"""Centralized user-facing strings (i18n-ready)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast


def _flatten(obj: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten nested dict to dot keys. All values are strings."""
    result: dict[str, str] = {}
    for key, value in obj.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten(cast(dict[str, Any], value), full_key))
        else:
            result[full_key] = str(value)
    return result


def _load_strings() -> dict[str, str]:
    from app.core.config import settings

    locale = getattr(settings, "locale", "en")
    root = Path(__file__).resolve().parent.parent.parent
    path = root / "locales" / f"{locale}.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return _flatten(cast(dict[str, Any], data))


_strings: dict[str, str] | None = None


def _get_strings() -> dict[str, str]:
    global _strings
    if _strings is None:
        _strings = _load_strings()
    return _strings


def get_message(key: str, **params: Any) -> str:
    """Return the message string for key, with placeholders substituted."""
    strings = _get_strings()
    text = strings.get(key)
    if text is None:
        return key
    try:
        return text.format(**params) if params else text
    except KeyError:
        return text


def get_error_code(key: str) -> str:
    """Return the last segment of the dot key (e.g. auth.user_exists -> user_exists)."""
    return key.split(".")[-1] if "." in key else key


def get_message_for_error(exc: Exception) -> tuple[str, str]:
    """Resolve (error_code, message) for API error responses.

    Falls back to (exc.error_code, str(exc)) for unrecognized exception types.
    """
    from app.services.auth_service import AuthenticationError
    from app.services.interest_service import InterestExtractionError

    error_code = getattr(exc, "error_code", None)
    if error_code is None:
        return ("error", str(exc))

    message_key = getattr(exc, "message_key", None)
    if message_key is not None:
        return (error_code, get_message(message_key))

    if isinstance(exc, AuthenticationError):
        key = f"auth.{error_code}"
    elif isinstance(exc, InterestExtractionError):
        key = f"interests.{error_code}"
    else:
        return (error_code, str(exc))

    return (get_error_code(key), get_message(key))
