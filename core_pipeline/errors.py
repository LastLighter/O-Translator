"""Typed application errors with user-safe messages."""

from __future__ import annotations

from typing import Any


class OTranslatorError(Exception):
    """Base error exposed by the API as a structured response."""

    code = "o_translator_error"
    status_code = 500

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        user_message: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.user_message = user_message or message
        self.details = details or {}

    def to_payload(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.user_message,
            "details": self.details,
        }


class InvalidImageError(OTranslatorError):
    code = "invalid_image"
    status_code = 400


class UploadTooLargeError(OTranslatorError):
    code = "upload_too_large"
    status_code = 413


class ConfigurationError(OTranslatorError):
    code = "configuration_error"
    status_code = 500


class OcrUnavailableError(OTranslatorError):
    code = "ocr_unavailable"
    status_code = 503


class OcrProcessingError(OTranslatorError):
    code = "ocr_failed"
    status_code = 422


class InpaintingUnavailableError(OTranslatorError):
    code = "inpainting_unavailable"
    status_code = 503


class InpaintingError(OTranslatorError):
    code = "inpainting_failed"
    status_code = 500


class TranslationError(OTranslatorError):
    code = "translation_failed"
    status_code = 502


class TranslationAuthError(TranslationError):
    code = "translation_auth_failed"
    status_code = 502


class TranslationTimeoutError(TranslationError):
    code = "translation_timeout"
    status_code = 504


class TranslationFormatError(TranslationError):
    code = "translation_invalid_json"
    status_code = 502


class JobNotFoundError(OTranslatorError):
    code = "job_not_found"
    status_code = 404


class QueueFullError(OTranslatorError):
    code = "queue_full"
    status_code = 503
