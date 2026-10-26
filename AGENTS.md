# AGENTS.md

本文件是本仓库的协作规则，面向 AI 代理和人类贡献者。开始修改前先读完；
涉及构建、打包或 `dist/` 的任务，必须执行第 2 节的完整性验收。

## 1. 项目速览

O-Translator 是本地优先的截图翻译工具，主流程为：

```text
截图 → RapidOCR（PP-OCR ONNX）→ LaMa/OpenCV 去字 → DeepSeek 批量翻译 → Pillow 自适应排版
```

- `backend/`：FastAPI 本地服务（`launcher.py --backend` 或 `backend.main:run`）。
- `core_pipeline/`：OCR、LaMa/OpenCV 去字、翻译、排版与运行时 provider 策略。
- `client/`：PySide6 桌面端（托盘、热键、框选、HUD）和浏览器工作台。
- `scripts/`：构建、模型下载、GPU runtime 打包、发布校验。
- `models/`：ONNX 权重，体积大且已 gitignore，不入库。
- `dist/`：PyInstaller 产物和发布 zip；不要在这里维护源码。

## 2. 硬性规则：dist 产物必须完整

**任何留在 `dist/` 里、准备交付或已经生成的包/文件夹，都必须自带运行所需的全部文件，
不能依赖用户另行补装、另行解压或联网下载。** 每个完整发行包至少包含：

| 必需项 | 位置 / 判定 |
|---|---|
| 可执行程序 | `O-Translator.exe`（> 1 MB） |
| Python 运行时 | `_internal/` 完整目录 |
| LaMa 修复模型 | `_internal/models/lama/*.onnx` 或可写的 `models/lama/*.onnx`，≥ 20 MB（正常约 196–208 MB） |
| OCR 模型 | `_internal/models/rapidocr/*.onnx` 和/或 `_internal/rapidocr/models/*.onnx`，至少含 det + rec |
| CUDA 执行环境 | `gpu_runtime/onnxruntime/capi/onnxruntime_providers_cuda.dll` + `cudnn*_9.dll` + `cublas*_12.dll`/`cudart*_12.dll` 等 CUDA 12 DLL |
| 使用说明 | `使用说明.txt`（由 `packaging/README-Windows.txt` 复制） |

细则：

- **默认必须带 CUDA runtime。** 确实要交付 CPU-only 版本时，必须显式使用
  `verify_dist.py --cpu-only` 验收，并在文件名或说明中标注 `cpu`，不能默默省略。
- 单独的 `gpu_runtime/` 加速包（`dist/O-Translator-gpu-runtime-cuda12.zip`）只是
  add-on，**不算完整发行包**；完整包必须同时含 LaMa、OCR 和 `gpu_runtime/`。
- 每个 `dist/` 产物要么补齐并通过验收，要么明确标记为旧版/半成品并清理；
  不要把缺文件的目录或 zip 当作“已完成”留在发布目录。
- 打包用 `scripts/package_zip.py`（正斜杠条目名，保留空目录），不要用
  PowerShell `Compress-Archive` 直接压 `dist`。
- 打包前先关闭正在运行的 `dist\...\O-Translator.exe`，否则
  `data\desktop-backend.log` 会锁住文件（PyInstaller 报 `WinError 32`）。

## 3. 完整性验收（打包完成的唯一标准）

```powershell
# 解压后的完整包（默认要求 CUDA runtime）
python scripts/verify_dist.py dist\O-Translator

# 发布 zip（无需解压，直接读中央目录）
python scripts/verify_dist.py dist\O-Translator-windows-x64.zip

# 明确交付 CPU-only 版本时才允许
python scripts/verify_dist.py dist\O-Translator-cpu --cpu-only
```

命令返回 0 且结果为 `PASS` 才能说“打包完成”。脚本会检查 exe、`_internal/`、
LaMa、RapidOCR det+rec、CUDA provider/cuDNN/cuBLAS DLL 和 `使用说明.txt`，
并把结果按 `PASS` / `WARN` / `FAIL` 列出。`WARN` 不阻塞，但要在交付说明中注明
（例如旧 PaddleOCR 构建）。

## 4. 冻结路径与模型查找

- `BASE_DIR` / `_app_root()`：exe 同级目录，**可写**；放 `data/`、`models/`。
- `BUNDLE_DIR` / `_bundle_root()`：PyInstaller 的 `_MEIPASS`（即 `_internal/`），
  **只读**；放内置 LaMa、RapidOCR 权重、网页与 QSS 资源。
- LaMa 查找：先可写的 `models/lama/`，再内置的 `_internal/models/lama/`；
  按 `OTR_LAMA_PRECISION` 选择 `_sim`/`_fp16`/`_int8`/`_optimized` 变体，
  最后回退原图。`OTR_LAMA_MODEL_PATH` 可显式覆盖。
- OCR：`_internal/models/rapidocr/` 的 PP-OCRv4/v5 权重在首次启动时复制到可写的
  `models/rapidocr/`；PP-OCRv6 small 在 `_internal/rapidocr/models/` 内随 wheel 携带。
- GPU：`gpu_runtime/` 必须与 `O-Translator.exe` 同级（或由 `OTR_GPU_RUNTIME_DIR`
  指定）；`activate_gpu_runtime()` 会在首次 `import onnxruntime` 前把它放到
  `sys.path` 和 Windows DLL 搜索路径，失败自动回退 CPU。

## 5. 常用命令

```powershell
# 开发环境
pip install -r requirements.txt -r requirements-desktop.txt
python -m pytest
ruff check .
python launcher.py --doctor

# 构建桌面端（会校验内置 LaMa 与 RapidOCR）
powershell -ExecutionPolicy Bypass -File scripts/build_windows_exe.ps1

# 生成 CUDA add-on（约 2 GB）
python scripts/build_gpu_runtime.py

# 重新打包一个已运行过的目录（排除日志和运行时下载的模型）
python scripts/package_zip.py dist\O-Translator dist\O-Translator-windows-x64.zip `
    --exclude data --exclude models
```

`O-Translator.exe --doctor` 会打印 `lama_available`、`lama_model_source`、
`ort_active`、可用 provider 和 RapidOCR 初始化结果，是排查打包环境的首选入口。

## 6. 工程约定

- 模型权重不入库；下载用 `scripts/download_models.py --lama/--ocr`。
- 不写死 API Key；只从 `DEEPSEEK_API_KEY` / `OTR_DEEPSEEK_API_KEY` 读取。
- 新增依赖时同步更新 `requirements.txt`（以及适用的 `requirements-desktop.txt`、
  `pyproject.toml`）。
- 改动打包路径、spec 或 `backend/config.py` 的搜索顺序后，必须重新跑构建和
  `scripts/verify_dist.py`，不能只跑单元测试。
- 提交前跑 `pytest` 和 `ruff check .`；UI 改动至少做一次 `launcher.py --doctor`
  或启动冒烟。
