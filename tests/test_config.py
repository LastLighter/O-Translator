from __future__ import annotations

from backend.config import Settings


def test_deepseek_api_key_alias(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    settings = Settings(_env_file=None, translation_provider="auto")
    assert settings.deepseek_api_key == "test-key"
    assert settings.effective_translation_provider == "deepseek"
    assert settings.translation_configured is True


def test_auto_provider_without_key_uses_echo(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OTR_DEEPSEEK_API_KEY", raising=False)
    settings = Settings(_env_file=None, translation_provider="auto")
    assert settings.effective_translation_provider == "echo"
    assert settings.translation_configured is False


def test_relative_lama_path_is_resolved_under_project():
    settings = Settings(_env_file=None, lama_model_path="models/lama/demo.onnx")
    assert settings.resolved_lama_path.is_absolute()
    assert settings.resolved_lama_path.as_posix().endswith("models/lama/demo.onnx")


def test_bundled_optimized_model_is_found_without_download(monkeypatch, tmp_path):
    import backend.config as config

    writable = tmp_path / "app"
    requested = writable / "models" / "lama" / "lama_fp32.onnx"
    bundle = tmp_path / "_internal"
    lama_dir = bundle / "models" / "lama"
    lama_dir.mkdir(parents=True)
    folded = lama_dir / "lama_fp32_sim.onnx"
    folded.write_bytes(b"folded")
    monkeypatch.setattr(config, "BUNDLE_DIR", bundle)
    monkeypatch.setattr(config, "DEFAULT_LAMA_MODEL_PATH", requested)
    monkeypatch.setattr(config, "MIN_LAMA_BYTES", 1)

    settings = config.Settings(_env_file=None, lama_model_path=requested)
    assert settings.resolved_lama_path == folded
    assert settings.lama_model_available is True
    assert settings.lama_model_source == "bundled"


def test_explicit_lama_path_wins_over_optimized_sibling(monkeypatch, tmp_path):
    import backend.config as config

    monkeypatch.setattr(config, "MIN_LAMA_BYTES", 1)
    explicit = tmp_path / "custom.onnx"
    explicit.write_bytes(b"custom")
    sibling = tmp_path / "custom_sim.onnx"
    sibling.write_bytes(b"sim")
    settings = config.Settings(_env_file=None, lama_model_path=explicit)
    assert settings.resolved_lama_path == explicit


def test_invalid_pipeline_options_are_normalized():
    from core_pipeline.types import PipelineOptions

    options = PipelineOptions(
        source_lang="JP",
        target_lang="zh-cn",
        dilate_px=99,
        font_scale=9,
    )
    assert options.source_lang == "ja"
    assert options.target_lang == "zh"
    assert options.dilate_px == 7
    assert options.font_scale == 1.6