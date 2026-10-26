# 性能优化报告：OCR、LaMa 与嵌字

> 目标：缩短一次截图翻译的端到端等待时间，重点排查 OCR 与嵌字（typesetting / render）两个阶段。
>
> 结论先说：**嵌字在真实页面上通常只有几十毫秒（本次样本 12–66 ms；60 个区域的密集页优化前可达 1.3 秒），真正的瓶颈是 OCR 和 LaMa。** OCR 的 6–15 秒里约 8 秒是首次模型初始化；LaMa 的 25–35 秒冷启动则来自 17k 节点动态图的 ONNX 图优化。本轮把 OCR 冷启动移到服务启动的后台预热，消除了 `auto` 模式重复跑整页 OCR，把密集页嵌字提速约 5 倍；同时用 `onnxsim` 把 LaMa 图从 17,480 折叠到 3,028 节点，加载从 32–35 秒降到 1.6–2.3 秒，并给 ORT 加了 4 线程上限。优化后稳态 OCR 约 0.8–1.6 秒，嵌字 12–66 ms，LaMa 推理 2.6 秒（空闲）/约 7 秒（有后台负载）。

## 1. 测试环境

- Windows 11，Intel Core i7-13620H（10 核 / 16 线程），16 GB 内存
- Python 3.13.5，PaddleOCR 3.3.1 / PaddlePaddle 3.2.1，Pillow 12.1.1，OpenCV 4.10，onnxruntime 1.31
- 样本：`examples/demo_manga.png`（900×1280）、`examples/demo_game_ui.png`（1400×900），以及 `data/jobs/` 中 7 个真实任务的 `original.png`
- 除冷启动测试外，所有计时都在模型已预热后取 3 次中位数；CPU 推理对后台负载非常敏感（本次机器上同时有浏览器、游戏客户端等），同一脚本在不同时刻的绝对值可相差约 2 倍，因此**对比数字主要看同一脚本内背靠背测得的相对提升**

## 2. 优化前后对比

### 2.1 OCR：稳态单页

`source_lang=ja`、`mode=manga`，预热后 3 次中位数：

| 页面 | 优化前 | 优化后 | 提升 |
|---|---:|---:|---:|
| `demo_game_ui.png`（1400×900，12 区域） | 3303 ms | 912 ms | **3.6×** |
| `demo_manga.png`（900×1280，3 区域） | 1103 ms | 939 ms | 1.2× |
| 真实任务 094957（475×705，10 区域） | 1858 ms | 1837 ms | 1.0×（保留了两处有效精修） |
| 真实任务 080554（900×1280，3 区域） | 1065 ms | 909 ms | 1.2× |

> 表中的绝对值是在机器较空闲时测得；复现时如果后台有其他程序，绝对值可能翻倍，但同一脚本内“旧/新”背靠背对比的相对提升不变。

`source_lang=auto`（浏览器默认）曾经会把整页 OCR **跑两遍**（中文模型一遍 + 日文模型一遍）：

| 页面 | 旧 auto（两遍整页） | 新 auto（原地精修） | 提升 |
|---|---:|---:|---:|
| `demo_manga.png` | 1385 ms | 769 ms | **1.8×** |
| 真实任务 094957 | 2703 ms | 1756 ms | **1.5×** |
| 真实任务 080554 | 1851 ms | 992 ms | **1.9×** |

### 2.2 OCR：冷启动（首个任务）

| 场景 | 优化前 | 优化后 |
|---|---:|---:|
| 服务启动后的第一次 OCR | 8155 ms | 931 ms |
| 启动时后台预热耗时 | — | 7640 ms（不阻塞 UI） |

### 2.3 嵌字（render）

| 场景 | 优化前 | 优化后 | 提升 |
|---|---:|---:|---:|
| 合成密集页，60 个区域 | 1297 ms | 243 ms | **5.3×** |
| 真实任务回放，3–12 个区域（同一份代码内模拟旧路径） | 24–42 ms | 12–22 ms | 1.5–2.3× |
| 保存的 094957 任务元数据 | 314.6 ms | 21–30 ms（回放） | 10× 量级（含旧版冷字体缓存/机器负载，仅作参考） |

优化后嵌字在真实页面里通常只有 **12–35 ms**（后台负载高时可到 60 ms 左右），已经可以忽略。

### 2.4 端到端（关闭翻译网络请求，LaMa 已预热）

| 页面 | 总计 | OCR | 背景修复 | 蒙版 | 嵌字 |
|---|---:|---:|---:|---:|---:|
| 真实任务 094957（8 区域） | 5583 ms | 2637 ms | **2892 ms** | 2.8 ms | 33 ms |
| `demo_manga.png`（3 区域） | 3688 ms | 1007 ms | **2644 ms** | 1.7 ms | 16 ms |
| `demo_game_ui.png`（12 区域） | 4412 ms | 1526 ms | **2745 ms** | 3.3 ms | 33 ms |

可见本轮之后，**背景修复（LaMa）已经成为剩余的最大头**，第 6 节给出下一步建议。端到端表里各阶段取自一次较空闲时的代表性运行，OCR 列略高于 2.1 的单独基准（同一进程连续跑 3 次的负载差异）。在同时开着浏览器、游戏客户端时，LaMa 推理会从 2.6 s 涨到 7–8 s，端到端约 9–11 s；这是 CPU 争抢，不是新代码回退。

### 2.5 LaMa：为什么慢，以及本轮又做了什么

先纠正一个配置误解：当前 `Carve/LaMa-ONNX` 导出的输入形状是**固定 512×512**（`image [batch,3,512,512]`、`mask [batch,1,512,512]`），所以 `OTR_INPAINT_MAX_SIDE` 对它不起作用——每次推理都是 512×512，和原图大小无关。

慢的原因：

- 模型本身是 FFC（Fourier Convolution）LaMa，198 MB fp32；ONNX 图有 **17,480 个节点**，其中 7,054 个 Constant、1,188 个 Shape、1,348 个 Reshape、1,322 个 Slice、216 个 Einsum、144 个 Sin/Cos。导出时 FFT 被展开成大量基础算子，CPU 上执行效率很低。
- 推理本身在较空闲时约 **2.6 s**；后台有其他程序时会被线程争抢放大到 **7–13 s**。
- 冷启动的 25–35 s 主要不是推理，而是 ONNX Runtime 对这张 17k 节点图做图优化/常量折叠；首次创建 session 时一次付清。

本轮的两项改动：

| 指标 | 优化前 | 优化后 | 说明 |
|---|---:|---:|---|
| 图加载（冷启动） | 32–35 s | **1.6–2.3 s** | `onnxsim` 固定 batch/shape 后节点 17,480 → 3,028 |
| 单次推理（同一负载） | 12.9 s | **7.0 s** | ORT 线程从“全部逻辑核”改为 4 线程 |
| 首次 LaMa（加载 + 预热 + 首次推理） | ~45 s | **~8.5 s** | 上两项叠加 |
| 首个任务（OCR + LaMa 冷启动） | ~33–43 s | **~9–10 s** | 假设任务在后台预热完成后到达 |
| 输出差异 | — | 最大 1/255，>1 的像素 0 个 | `--verify` 实测 |

- **图固定化**：新增 `scripts/optimize_lama.py`，用 `onnxsim` 把动态 shape 相关节点折叠掉，生成 `lama_fp32_sim.onnx`；inpainter 自动优先加载它。`onnxsim` 不是硬依赖，未安装时继续用原图。
- **线程数**：ORT 默认拿满 16 个逻辑核，但 FFC/FFT 子图在 4 线程后基本不再扩展，反而和 OCR/浏览器抢核。实测 1/2/4/8/12/16 线程分别为 14.8/7.5/6.8/10.7/15.1/20.4 s（同一负载，数字随机器波动），因此默认 `OTR_LAMA_THREADS=0`（自动取 `min(4, cpu//2)`），可显式覆盖。

## 3. 具体改动

### 3.1 OCR 语言后端复用（避免同一套权重加载两遍）

`ch`、`japan`、`chinese_cht` 在默认 `auto` profile 下选择的是同一对模型（`PP-OCRv5_mobile_det` + `PP-OCRv5_mobile_rec`），旧实现却按语言各建一个 `PaddleOCRBackend`，第二个语言首次使用时要再付一次模型初始化成本。

- 新增纯函数 `select_paddle_models()`，把“语言 → 模型名”的选择从实例里抽出来。
- 新增 `PaddleOCRBackend.model_signature()`。
- `LanguageAwareOCRBackend._backend()` 按模型签名去重：签名相同的语言共用一个已初始化的后端，并在 `read_language()` 里显式传入该语言的精修开关（不再修改 `self.lang`，避免多任务竞争）。

文件：`core_pipeline/ocr.py`

### 3.2 `auto` 模式原地精修（取消第二次整页 OCR）

旧逻辑：`source_lang=auto` 先跑中文模型，发现假名后再用日文后端 **完整重跑** 检测 + 逐行识别。但两者的模型相同，日文后端唯一多做的事是“对可疑的水平框做放大重检测”。

- 新增 `PaddleOCRBackend.refine_regions()`：只对已识别区域执行放大精修，复用整页检测和识别结果。
- `LanguageAwareOCRBackend` 在模型签名一致时走原地精修，只有韩语等模型不同的语言才退回完整重跑。
- 空页面直接复用结果，不再白跑第二遍。

效果：`auto` 模式从“两遍整页”降到“一遍整页 + 少量小图检测”，见 2.1 表。

### 3.3 收紧精修启发式（不再对高置信度 UI 文本做 100–200 ms 的无用检测）

旧 `_needs_cluster_refine()` 只要“文本短且宽”或“高度 ≤ 14 px”就触发精修。UI 截图里的 `23:48`、`CHAPTER IV`、`设置` 全部命中，每个白花约 200 ms；在 1400×900 的游戏 UI 上 7 个候选 **0 个真正拆分**，却消耗 1.6 秒。

新规则加入置信度门槛：

- 短而宽：`confidence < 0.92`（合并标题通常来自花体/描边，分数偏低；清晰 UI 标签接近 1.0）
- 极矮行：`confidence < 0.85`
- 低置信度中等行：维持 `< 0.70`

在真实漫画任务 094957 上，两处真正有用的精修（标题与 11 px 小字幕，置信度 0.82 / 0.50）都保留；UI 页从 7 次精修降到 0 次。

### 3.4 启动预热（把 8 秒冷启动移出首个任务）

- `PaddleOCRBackend.warm_up()`：加载检测/识别模型，并各跑一次极小推理，同时初始化精修用的第二个检测器。
- `LanguageAwareOCRBackend.warm_up()` 预热默认语言（日文场景顺带预热精修路径，共用后端时不会重复加载）。
- `AutoOCRBackend.warm_up()` 转发给第一个可用后端（默认 `OTR_OCR_ENGINE=auto` 走的正是这条链路）。
- `backend/main.py` 的启动线程先预热 OCR，再按原有逻辑预热 LaMa；预热不阻塞 HTTP 服务和 UI。
- 顺手修了 LaMa 的重复初始化竞争：预热线程与恰好到达的任务可能同时创建 ONNX session（198 MB 模型要重新做图优化）。现在 `_load()` 是单飞（single-flight）的，只会创建一次。

效果：首个任务 OCR 从 8155 ms 降到 931 ms（若任务恰好在预热完成前到达，仍可能看到部分初始化等待，但不会比旧版更差）。LaMa 首次初始化的 25–35 秒来自 17k 节点动态图的 ONNX 图优化，3.7 的固定形状折叠把它降到 1.6–2.3 秒；3.8 再把推理线程从“占满所有核”收到 4 个。

### 3.5 嵌字：整页像素只转换一次

旧 `Renderer.render()` 对每个文本框调用 `analyze_text_style()`，后者把 **整页** 转成 RGB numpy 数组；随后又调用 `extract_colors()`，再转换一次。一个有 N 个框的页面会做 2N 次整页拷贝。

- `Renderer.render()` 在循环外计算一次 `source_pixels = np.asarray(original.convert("RGB"))`。
- `analyze_text_style(..., pixels=)` 与 `extract_colors(..., pixels=)` 接受这份缓存，按框切片（numpy view，不再拷贝）。

输出颜色与旧实现逐像素一致（新增测试覆盖）。

### 3.6 嵌字：CJK 快速换行（保持像素级一致）

字体自适应要在大约 `start → 7` 的每个字号上完整跑一次 `wrap_text`。旧实现对每个 token 都调用 `font.getlength(整行候选)`，CJK 行长 k 时是 O(k²) 次字形布局，是整个嵌字阶段的主要成本。

- 对“纯 CJK/假名/全角标点”的段落，先按 **单字 advance** 求和做贪心换行（每个字号只测每个唯一字形一次）；换行规则（禁则、开闭标点）与原实现逐条对应。
- 换行完成后用 `getlength(整行)` 校验加性宽度；若字体存在字距调整导致不一致，自动回退到原来的精确换行。
- `_text_width` 增加 `(font, text)` 的 LRU 缓存；段落类型判断也做了缓存。

字体选择结果与旧算法完全一致：400 组随机样本（文本、框尺寸、字号范围、字重、横/竖排、`max_size`）逐一对比，**400/400 相同**。

### 3.7 LaMa：固定形状图折叠（加载 15–20×）

`scripts/optimize_lama.py` 读取 ONNX 输入形状，把动态 batch/空间维度固定为实际值（512×512），交给 `onnxsim` 做常量折叠和形状推断；输出 `lama_fp32_sim.onnx`，原图保留。

- 节点数 17,480 → 3,028，文件 195.6 MB。
- session 加载 32–35 s → **1.6–2.3 s**；推理基本不变（1.02×）。
- 数值验证：最大绝对误差 6.9e-4（`--verify`），量化成 uint8 后最大差 1、均值 0.000、没有像素差 >1。
- `LamaOnnxInpainter.optimized_path()` 自动优先 `_sim`，加载失败会回退到原图。

### 3.8 LaMa：ORT 线程数上限

`ONNX Runtime` 默认用满所有逻辑核。对这张 FFT/Einsum 图，16 线程在混合负载下反而比 4 线程慢约 1.8×。现在 `SessionOptions.intra_op_num_threads=4`（自动，最多 4）、`inter_op_num_threads=1`，并用 `OTR_LAMA_THREADS` 暴露覆盖。

### 3.9 OCR：完全相同页面的结果缓存

`LanguageAwareOCRBackend` 增加 8 项的精确页面缓存（blake2b 内容摘要 + 尺寸/模式/语言）。同一张图重复提交时 OCR 直接复用；首张图不受影响。

- 端到端实测：同一张 `demo_manga.png` 第二次提交，`ocr_ms` 从 **1536 ms → 12.8 ms**。
- 缓存返回深拷贝，后续 `order_regions()` 修改 id 不会污染缓存；一个像素不同就会 miss。
- 对桌面区域监控的“画面几乎没变”场景帮助有限（监控本身会跳过未变化帧），但对重复点“翻译”/网页重发有效。

## 4. 验证与回归

- `python -m pytest -q`：全部通过（含新增用例）。
- `python -m ruff check backend core_pipeline client tests scripts`：通过。
- 新增测试：
  - `test_fast_cjk_wrap_matches_generic_wrapper`：快速换行与精确换行逐行一致
  - `test_fit_returns_the_largest_fitting_size`：自适应字号仍取“最大可容纳字号”
  - `test_cached_page_pixels_match_on_the_fly_conversion`：缓存整页像素不改变颜色分析
  - `test_language_aware_ocr_shares_backends_with_identical_models`
  - `test_auto_japanese_refine_reuses_the_recognized_page`
  - `test_refine_heuristic_ignores_confident_ui_labels`
  - `test_auto_ocr_warms_the_first_available_backend`
  - `test_language_aware_ocr_caches_identical_pages`
  - `test_lama_prefers_the_simplified_sibling`
  - `test_lama_default_threads_are_capped`
- LaMa 数值一致性：`scripts/optimize_lama.py --verify` 输出最大误差 5.6e-4，`allclose(1e-3)=True`；整图 inpaint 输出 uint8 最大差 1、均值 0.000、没有像素差 >1。`test_lama_falls_back_when_the_simplified_graph_is_broken` 覆盖损坏 `_sim` 自动回退原图的路径。
- 单独的一致性脚本（一次性验证，未入库）对旧线性扫描与当前 `fit_text` 做了 400 组随机对比，字号、行拆分、`width/height`、`overflow` 全部一致。

## 5. 尝试过但没有采用的方案

这些是“看起来能提速、实测更慢或会改变输出”的路径，记录下来避免重复踩坑：

1. **识别批处理**：`TextRecognition.predict(list_of_crops, batch_size=N)` 在 CPU 上反而更慢。整批（按最大宽高比补齐）实测 0.67–0.85×；进一步按宽高比 2%/5%/10%/20% 分桶（批大小 ≤8）仍在 **0.86–1.14×** 之间，且没有稳定收益。原因是 Paddle 的识别模型计算量本身占主导，批处理节省的调用开销抵不过补齐浪费。未采用。
2. **PaddleOCR 高层 pipeline**：用 mobile det/rec + `text_recognition_batch_size=8` 直接 `PaddleOCR.predict()`，在 900×1280 和 1400×900 上比现有“检测 + 逐行识别”慢 **30–50%**（1343 vs 891 ms、1795 vs 1323 ms），还会漏掉 `CHAPTER IV` 等文本。未采用。
3. **检测批处理**：不同尺寸的裁剪图直接报 `all input arrays must have the same shape`；同尺寸批处理也不比逐张快（83 ms vs 44 ms）。未采用。
4. **降低检测分辨率**：`limit_side_len=960/1080` 在 900×1280 漫画上分别丢失 1/2 条文本，在 UI 上丢失 `CHAPTER IV`；耗时没有变快（因为小图本来就没有被 downscale）。保留 1280。
5. **OCR 线程数**：`cpu_threads=4` 在 475×705 上比默认 10 快约 9%，但在 900×1280/1400×900 上基本持平或略慢（1309 vs 1301 ms、1609 vs 1575 ms）。没有足够稳定的收益，不修改默认值。
6. **字号二分查找**：CJK 禁则会让“是否溢出”随字号非单调。实测存在 `12px 可以、13–19px 溢出、20–25px 又可以、26px 起溢出` 的情况；二分会把字号错误地压到 12px（旧实现会选 25px）。因此保留自顶向下的线性扫描，改用 3.6 的快速换行把每次扫描变便宜。
7. **降低精修放大倍率**：能再省一点，但会改变小字幕的拆分质量，样本不足，暂不调整。
8. **LaMa 局部裁剪（早期结论）**：当时按每个文本包围盒各跑一次 LaMa，模型输入固定 512×512，单次推理时间不变，N 个框就变成 N 倍成本，因此没有采用。第 7 节的改造把策略改成“简单背景直接跳过 LaMa；复杂背景合并为**一个** union ROI”，才让裁剪真正有收益。

## 6. 当前瓶颈与下一步

按 2.4 的端到端数据，优化后的大头是：

1. **LaMa 推理 2.6 s（空闲）/ 7 s（有后台负载）**：这是 198 MB FFC 模型在 CPU 上跑 512×512 的固有成本。图加载已经从 32–35 s 降到 1.6–2.3 s，推理本身没有“配置开关”能大幅降低。
   - 有 NVIDIA GPU 时用 `onnxruntime-gpu` / `paddlepaddle-gpu`；OpenVINO/TensorRT execution provider 也可能带来 2× 以上收益。
   - 换更小或更快的修复模型（例如 256×256 导出、LaMa-FCF 裁剪版）是根本解法，但需要重新导出/验证质量。
   - 纯 UI/纯色背景现在由 `AutoInpainter` 自动识别并跳过 LaMa（σ < 24），无需手动改 `OTR_INPAINT_ENGINE`；第 7 节实测 1400×900 UI 页快速修复 34 ms。
   - 用 ONNX Runtime 的 `optimized_model_filepath` 缓存图优化结果，可进一步缩短**后续进程启动**的初始化（首次仍需完整优化）。
2. **OCR 0.9–1.6 s**：整页检测 0.4–0.55 s + 逐行识别 45–55 ms/行，已经接近 `PP-OCRv5_mobile_*` 在这台 CPU 上的下限。
   - 第 5 节列出的批处理、pipeline、降分辨率、线程数都没有稳定收益。
   - 再往下需要换推理后端（GPU / OpenVINO / ONNX Runtime HPI）或接受更低的识别精度。
   - 完全相同的页面重复提交会命中 3.9 的缓存（OCR ~13 ms）；区域监控若要做“局部变化只识别变化块”，需要在客户端传 diff 区域，工作量较大。
3. **翻译网络请求 1–2.6 s**：已有分块并发，进一步可做跨任务缓存（相同文本直接复用）。
4. **桌面区域监控**：同一区域连续截图经常只有轻微变化，可在客户端按图像哈希缓存上一次结果，命中时直接显示，跳过整条流水线。

## 7. 2026-10 改造：自适应快速修复、ROI LaMa 与两段式渲染

本轮针对“高频截图翻译”重构了背景修复与客户端渲染：

### 9.1 自适应快慢路径

- 对每个文本蒙版连通域，在蒙版外圈采样背景像素，计算通道标准差的均值 σ。
- `σ < OTR_INPAINT_SOLID_STDDEV`（默认 3）：纯色中值填充；
- `σ < OTR_INPAINT_FAST_STDDEV`（默认 24）：局部 `cv2.inpaint(TELEA)`，不加载/不调用 LaMa；
- 否则：只把 `bbox + 16px` 的 ROI 裁剪送 LaMa，再按蒙版 alpha/`seamlessClone` 融合回主画布。

实测（i7-13620H，同一进程预热后，`AutoInpainter`）：

| 场景 | 区域 | 快速路径 | LaMa ROI | 说明 |
|---|---:|---:|---:|---|
| 合成 UI（1280×720，12 个纯色框） | 12 simple / 0 complex | **17–34 ms** | 0 ms | 完全跳过 LaMa，端到端 `inpaint_fast_ms` 34 ms |
| 真实漫画任务 094957（475×705） | 0 simple / 3 complex | 10–18 ms | **3.0–3.4 s**（1 个 union ROI） | 3 个独立 patch 曾需 10–16 s，改为默认 1 个 union ROI |
| 同页全图 LaMa | — | — | 5.3–6.9 s | ROI 裁剪仍更快，因为只跑一次 512×512 |
| `demo_game_ui.png`（1400×900，12 区域） | 9 simple / 0 complex | **34 ms** | 0 ms | 阈值 24 时该页全部走快速路径 |

> 关键结论：LaMa 的推理成本由固定的 512×512 输入决定，**多裁几个 patch 不会更快**，反而成倍增加。默认 `OTR_INPAINT_MAX_ROI_PATCHES=1` 把所有复杂区域合并为一个 union ROI；需要区域独立性时可调大。

### 9.2 两段式渲染

- 后端先算快速画布；翻译完成后立即渲染 pass 1，写入 `preview.png` 并通过 job 记录的 `preview` 字段发布；
- LaMa 在后台继续跑 ROI，完成后渲染 pass 2，写入 `result.png`；
- 桌面客户端轮询间隔从 800 ms 降到 300 ms，拿到 preview 立即显示 HUD，最终结果用 180 ms 淡入替换；
- HTTP 客户端新增的下载/JSON 开销可忽略。

实测 `demo_game_ui.png`（`translate=false`，OCR 已预热）：

| 指标 | 数值 |
|---|---:|
| 解码 + OCR + 蒙版 | 15 + 1569 + 2 ms |
| 快速修复 | 34 ms |
| pass-1 渲染 | 53 ms |
| **pass-1 可见（preview_ready_ms）** | **1675 ms** |
| 最终渲染 | 44 ms |
| **端到端** | **1727 ms** |
| LaMa 调用 | 0（全部快速路径） |

同一页在旧阈值 12 时有一个 σ=23.3 的区域被判为复杂，触发一次 LaMa ROI：

| 指标 | 数值 |
|---|---:|
| pass-1 可见 | 1598 ms |
| LaMa ROI | 3.8 s |
| 端到端 | 5.4 s |

因此“是否调用 LaMa”是 UI/游戏场景能否稳定在 2 秒内的关键；漫画网点页仍主要受 LaMa 3–4 s 的 CPU 推理限制。要做亚秒级，需要 GPU/OpenVINO/DML 或更小的修复模型，而不是继续裁剪 ROI。

### 9.3 精度实验

`lama_fp32_fp16.onnx`（99 MB）和 `lama_fp32_int8.onnx`（55 MB）已可由
`scripts/optimize_lama.py --precision fp16|int8 --verify` 生成，但本机 CPU 实测：

| 变体 | 推理 | 最大误差 |
|---|---:|---:|
| FP32 folded | 3.0–4.2 s | 0.0005（仅图折叠） |
| FP16 | 5.6–6.2 s | ~204/255 |
| Dynamic INT8 | 5.4–41 s | ~163/255 |

两者都不可用于 CPU 默认路径，故打包只内置 FP32 folded 图；FP16/INT8 保留为
GPU provider 的可选项，启用前必须 `--verify`。

### 9.4 执行 provider

`core_pipeline/runtime.py` 统一 provider 顺序：

```
CUDAExecutionProvider -> OpenVINOExecutionProvider ->
DmlExecutionProvider -> CPUExecutionProvider
```

只使用已安装 ORT 变体实际暴露的 provider；OpenVINO 可配 `device_type`，
DirectML/CUDA 可配 device id；provider 初始化失败自动 CPU 重试。
`/api/health.runtime` 与 `/api/health.engines.inpainting.providers` 可核对实际选择。

### 9.5 打包

- `O-Translator.spec` 把 `models/lama/lama_fp32_sim.onnx` 放进
  `_internal/models/lama/`；
- `backend/config.py` 同时搜索 exe 目录（可写，下载目标）和 `_MEIPASS`
  （只读，内置模型），因此解压即用；
- 只有当所有候选（`_sim`/`_fp16`/`_int8`/原图）都不存在时才启动下载；
- 构建脚本会拒绝没有内置模型的产物，`smoke_test_exe.ps1` 会断言
  `lama_weights_present=true` 且 `download=null`。

### 7.6 GPU 验证（RTX 5060 Laptop / CUDA 12.8 驱动）

基础包仍只带 CPU ORT（CUDA 运行库约 2 GB，不适合所有人）。新增可选
`gpu_runtime/` 附加包：解压到 `O-Translator.exe` 同级后，
`core_pipeline/runtime.py` 会在任何 `import onnxruntime` 之前把它插到
`sys.path` 最前，并把 `gpu_runtime/onnxruntime/capi` 加入 DLL 搜索路径；
provider 策略因此自动选中 `CUDAExecutionProvider`，失败仍回退 CPU。

环境：`onnxruntime-gpu 1.24.4` + CUDA 12.6 用户态库 + cuDNN 9.5。

| 指标 | CPU ORT | CUDA ORT | 提升 |
|---|---:|---:|---:|
| LaMa 512×512 预热后单次 | ~2.1 s | **0.21–0.30 s** | ~8–10× |
| 漫画 475×705（3 个复杂区域，1 个 union ROI） | 3.0–3.4 s | **0.26 s** | ~12× |
| 端到端漫画（OCR + preview + LaMa） | ~4.8 s | **1.83 s** | ~2.6× |
| UI 1400×900（快速路径，无 LaMa） | ~1.7 s | 相同 | —（OCR 限制） |
| CUDA vs CPU 输出差异 | — | max 0.47/255，无像素 >1 | — |

关键点：

- `runtime.activate_gpu_runtime()` 必须在第一次 `import onnxruntime` 前执行；
  `launcher.py`、`backend/main.py` 已提前调用，避免 PaddleX 先导入 CPU ORT。
- 若 `onnxruntime` 已被导入，激活会记录警告并保持 CPU；重启进程即可。
- `OTR_DISABLE_GPU=1` 强制 CPU，`OTR_GPU_RUNTIME_DIR` 可指向自定义附加包。
- 本机 PaddleOCR 仍为 CPU：PaddlePaddle 没有适用于 Windows + Python 3.13 的
  GPU wheel；OCR（约 0.9–1.6 s）是剩余的主要延迟。

## 8. 复现（新增命令）

```powershell
# 生成 folded + FP16/INT8 变体（FP16/INT8 需要 --allow-lossy 才会保留）
python scripts/optimize_lama.py --verify
python scripts/optimize_lama.py --precision fp16 --verify --allow-lossy
python scripts/optimize_lama.py --precision int8 --verify --allow-lossy

# 可选 CUDA 运行库（约 2 GB）并打包 gpu_runtime 附加包
powershell -ExecutionPolicy Bypass -File scripts/setup_gpu_runtime.ps1
python scripts/build_gpu_runtime.py

# 验证 GPU provider 与 CPU/GPU 输出差异
python scripts/verify_gpu.py --require-gpu
python -c "from core_pipeline.runtime import describe_runtime; print(describe_runtime())"

# 观察自适应路径：改成任意截图/区域
python -c "from core_pipeline.inpainting import build_inpainter; ..."

# 回归
python -m pytest -q
python -m ruff check backend core_pipeline client tests scripts
```

## 9. 配置建议（单机桌面场景）

| 配置 | 建议 | 理由 |
|---|---|---|
| `OTR_OCR_ENGINE` | `paddle` 或 `auto` | 默认 `auto` 已经会优先 Paddle，并带有预热 |
| `OTR_PADDLE_PROFILE` | `fast` 或 `auto` | `auto` 已是 mobile 模型；`quality` 会明显变慢 |
| `source_lang` | 明确填 `ja` / `zh` | 可完全跳过语言回退逻辑；现代码下 `auto` 的回退也已经很便宜 |
| `OTR_LAMA_THREADS` | `0`（自动，最多 4） | FFT/Einsum 图在 4 线程后不再扩展；有独显/独占机器可显式调大 |
| `OTR_INPAINT_MAX_SIDE` | 保持 1024 | **对当前固定 512×512 的 LaMa 导出无效**；仅动态 shape 导出生效 |
| `OTR_INPAINT_FAST_STDDEV` | `24`（UI/游戏）/ `8–12`（漫画保守） | 低于阈值直接跳过 LaMa；越小越容易走 LaMa |
| `OTR_INPAINT_MAX_ROI_PATCHES` | `1` | LaMa 单次 512×512 成本固定，多 patch 会成倍变慢 |
| `OTR_INPAINT_ENGINE` | `auto`；纯 UI 可 `opencv` | `auto` 已对简单背景自动走快速路径；`opencv` 强制全图 Telea |
| `OTR_LAMA_PRECISION` | `auto`/`fp32` | CPU 上 FP16/INT8 实测更慢且误差大；GPU 可自行验证 |
| `OTR_LAMA_DEVICE` | `auto` | 有 CUDA/OpenVINO/DirectML 时自动加速；否则 CPU |
| `OTR_PROGRESSIVE_PREVIEW` | `true` | pass-1 先显示，避免等 LaMa 黑屏 |
| `OTR_QUEUE_WORKERS` | 单用户可设 `1` | 避免两路 CPU 推理互相争抢；批处理场景再调大 |
| `OTR_MAX_IMAGE_SIDE` | 2400（默认） | 更小的上限会同时降低 OCR 精度与耗时 |