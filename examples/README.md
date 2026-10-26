# Example screenshots

Generated deterministically by `python scripts/make_demo_assets.py`.

| File | Description |
|---|---|
| `demo_game_ui.png` | 1400×900 fantasy game dialogue HUD in Chinese |
| `demo_manga.png` | 900×1280 manga page with vertical speech bubble |

The web UI "载入示例" button uses `client/web/assets/demo.png`, which is the same
as `demo_game_ui.png`. Use these files to test OCR, inpainting, and layout:

```powershell
curl.exe -X POST http://127.0.0.1:8765/api/jobs `
  -F "file=@examples/demo_manga.png" `
  -F "source_lang=ja" `
  -F "target_lang=en" `
  -F "mode=manga"
```

Set `OTR_PADDLE_LANG=japan` and restart the server before testing Japanese OCR;
the default Paddle model is Chinese/English.
