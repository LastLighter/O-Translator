# 恢复说明（2026-10-26）

这个目录不是原始项目，是**从 DSH 会话日志逐文件重建**出来的版本。
原始目录 `D:\DSH-workspace\O-Translator` 在 2026-10-20 ~ 10-26 之间被永久删除（未进回收站），
详见 `_recovery/` 与桌面上的 `O-Translator-recovered\RECOVERY-REPORT.md`。

## 当前状态

| 项 | 结果 |
|---|---|
| 文件数 | 151（源码、测试、脚本、文档、配置） |
| 行数 | 约 32,000 |
| Python 语法校验 | 115 个 `.py` 全部通过 `ast.parse` |
| `python -m pytest`（排除 `tests/test_ocr.py`） | 约 133 项：**通过 130+，失败 1，跳过若干** |
| 唯一失败 | `tests/test_pipeline.py::test_pipeline_keeps_misread_icon_pixels` |

## 已知缺口（重要）

1. **`tests/test_ocr.py` 是旧版残留**：它 `import PaddleOCRBackend`，而最终版 `core_pipeline/ocr.py`
   已经改成 RapidOCR-only、没有这个类。收集阶段就会 ImportError，需要按新 API 重写。
2. **有 18 个文件是"较旧但完整"的版本**（删除前又被改过，改动只存在于日志的 edit 记录里，
   已尽力回放；回放失败的改动没补上）。逐文件的重建时间与行数见 `_recovery/manifest.json`，
   主要的有：
   - `core_pipeline\ocr.py` 2358 → 回放后 2757 行（删除前 2811）
   - `client\main.py` 1883 → 回放后 1928 行（与删除前一致）
   - `backend\config.py`、`core_pipeline\runtime.py`、`backend\jobs.py`、`README.md`、`docs\*`
     仍是 10-18 ~ 10-19 的版本
3. **完全没有的东西**（不是丢失，是本来就不入库/体积大）：
   - `models\` 下的 ONNX 权重（LaMa ~200 MB、RapidOCR det/rec）
   - `.venv-gpu` / `.venv-gpu-cuda` 虚拟环境
   - `dist\` 打包产物与 `O-Translator-windows-x64.zip`
   - `.env`（API Key），键名见 `.env.example`

## 想把它跑起来

```powershell
cd D:\DSH-workspace\O-Translator
pip install -r requirements.txt -r requirements-desktop.txt   # 本机 Python 3.13 已装有全部依赖
python scripts\download_models.py --lama --ocr                # 重新拉 ONNX 权重
python launcher.py --doctor                                   # 检查 LaMa / OCR / ORT provider
python -m pytest                                              # test_ocr.py 需先重写
```

打包仍按 `AGENTS.md` 第 2、3 节：`scripts\build_windows_exe.ps1`
→ `python scripts\verify_dist.py dist\O-Translator`，返回 0 且 `PASS` 才算完成。

## 为什么建议现在就 git init

原项目**没有 git 仓库**，所以这次事故里没有任何版本历史可用——这是最大的教训。
已经初始化了本地仓库并做了首次提交，建议立刻加远端：

```powershell
git remote add origin <你的私有仓库地址>
git push -u origin main
```

`models/`、`dist/`、`.venv*`、`data/` 已在 `.gitignore` 里，不会推上去。
