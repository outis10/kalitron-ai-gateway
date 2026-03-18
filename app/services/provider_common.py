import json

from app.core.errors import ProviderResponseError

ALLOWED_IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


def normalize_media_type(media_type: str) -> str:
    if media_type in ALLOWED_IMAGE_CONTENT_TYPES:
        return media_type
    if "png" in media_type:
        return "image/png"
    if "webp" in media_type:
        return "image/webp"
    return "image/jpeg"


def parse_json_response(text: str, error_message: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProviderResponseError(error_message) from exc
    if not isinstance(data, dict):
        raise ProviderResponseError(error_message)
    return data
