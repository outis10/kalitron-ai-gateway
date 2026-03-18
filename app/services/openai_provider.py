import asyncio
import base64
import logging

from openai import AsyncOpenAI

from app.core.config import settings
from app.core.errors import ProviderResponseError, UpstreamServiceError
from app.models.responses import OCRResult, VisionResult
from app.services.provider_common import normalize_media_type, parse_json_response

logger = logging.getLogger(__name__)

_EXTRACT_PROMPT = """Analyze this document image and extract all visible text.
Return ONLY a valid JSON object (no markdown, no explanation) with this exact structure:
{
  "raw_text": "<all text visible in the document as a single string>",
  "structured_fields": {
    "<field_name>": "<value>"
  },
  "confidence": <float between 0.0 and 1.0 representing extraction confidence>
}

For structured_fields, extract key-value pairs using lowercase English keys such as:
date, total, issuer, receipt_number, full_name, id_number, curp, expiry_date, date_of_birth, address, rfc, folio.
Only include fields that are actually visible in the document."""

_VISION_PROMPT_TEMPLATE = """Analyze this {document_type} document image for authenticity and potential fraud.

Examine:
- Physical integrity (tears, folds, unusual damage)
- Print quality and consistency
- Font uniformity and spacing
- Security features appropriate for this document type
- Signs of digital manipulation or alteration
- Whether the document structure matches the expected format for {document_type}

Return ONLY a valid JSON object (no markdown, no explanation) with this exact structure:
{{
  "is_authentic": <true or false>,
  "fraud_indicators": ["<indicator 1>", "<indicator 2>"],
  "authenticity_score": <float between 0.0 and 1.0>,
  "notes": "<brief observation about the document>"
}}

fraud_indicators should be an empty list [] if no issues are found.
Be conservative: only flag clear anomalies as fraud indicators."""


class OpenAIOCRService:
    def __init__(self, config=settings) -> None:
        self.settings = config
        self.client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)

    async def extract_text(self, image_bytes: bytes, media_type: str = "image/jpeg") -> OCRResult:
        validated_media_type = normalize_media_type(media_type)
        image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

        logger.debug("Sending image to OpenAI for OCR extraction (size=%d bytes)", len(image_bytes))
        response = await self._request(_EXTRACT_PROMPT, image_b64, validated_media_type, "OCR")
        raw_response = self._extract_text_block(
            response, "OCR provider returned an unexpected response shape."
        )
        data = parse_json_response(raw_response, "OCR provider returned invalid JSON.")

        return OCRResult(
            raw_text=data.get("raw_text", ""),
            structured_fields=data.get("structured_fields", {}),
            confidence=float(data.get("confidence", 0.5)),
        )

    async def _request(
        self,
        prompt: str,
        image_b64: str,
        media_type: str,
        operation_name: str,
    ):
        last_error: Exception | None = None
        for attempt in range(self.settings.OPENAI_MAX_RETRIES + 1):
            try:
                return await asyncio.wait_for(
                    self.client.responses.create(
                        model=self.settings.OPENAI_MODEL,
                        input=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": prompt},
                                    {
                                        "type": "input_image",
                                        "image_url": f"data:{media_type};base64,{image_b64}",
                                    },
                                ],
                            }
                        ],
                    ),
                    timeout=self.settings.OPENAI_TIMEOUT_SECONDS,
                )
            except Exception as exc:  # pragma: no cover
                last_error = exc
                logger.warning(
                    "OpenAI %s request failed on attempt %d: %s",
                    operation_name,
                    attempt + 1,
                    exc,
                )
                if attempt < self.settings.OPENAI_MAX_RETRIES:
                    await asyncio.sleep(0.25 * (attempt + 1))
        raise UpstreamServiceError(f"{operation_name} provider is unavailable.") from last_error

    @staticmethod
    def _extract_text_block(response, error_message: str) -> str:
        text = getattr(response, "output_text", "")
        if not text:
            raise ProviderResponseError(error_message)
        return text


class OpenAIVisionService:
    def __init__(self, config=settings) -> None:
        self.settings = config
        self.client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)

    async def analyze_document(
        self,
        image_bytes: bytes,
        document_type: str,
        media_type: str = "image/jpeg",
    ) -> VisionResult:
        validated_media_type = normalize_media_type(media_type)
        image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
        prompt = _VISION_PROMPT_TEMPLATE.format(document_type=document_type)

        logger.debug(
            "Sending image to OpenAI for vision analysis (document_type=%s, size=%d bytes)",
            document_type,
            len(image_bytes),
        )
        response = await self._request(prompt, image_b64, validated_media_type, "Vision")
        raw_response = self._extract_text_block(
            response,
            "Vision provider returned an unexpected response shape.",
        )
        data = parse_json_response(raw_response, "Vision provider returned invalid JSON.")

        return VisionResult(
            is_authentic=bool(data.get("is_authentic", False)),
            fraud_indicators=data.get("fraud_indicators", []),
            authenticity_score=float(data.get("authenticity_score", 0.5)),
            notes=data.get("notes", ""),
        )

    async def _request(
        self,
        prompt: str,
        image_b64: str,
        media_type: str,
        operation_name: str,
    ):
        last_error: Exception | None = None
        for attempt in range(self.settings.OPENAI_MAX_RETRIES + 1):
            try:
                return await asyncio.wait_for(
                    self.client.responses.create(
                        model=self.settings.OPENAI_MODEL,
                        input=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": prompt},
                                    {
                                        "type": "input_image",
                                        "image_url": f"data:{media_type};base64,{image_b64}",
                                    },
                                ],
                            }
                        ],
                    ),
                    timeout=self.settings.OPENAI_TIMEOUT_SECONDS,
                )
            except Exception as exc:  # pragma: no cover
                last_error = exc
                logger.warning(
                    "OpenAI %s request failed on attempt %d: %s",
                    operation_name,
                    attempt + 1,
                    exc,
                )
                if attempt < self.settings.OPENAI_MAX_RETRIES:
                    await asyncio.sleep(0.25 * (attempt + 1))
        raise UpstreamServiceError(f"{operation_name} provider is unavailable.") from last_error

    @staticmethod
    def _extract_text_block(response, error_message: str) -> str:
        text = getattr(response, "output_text", "")
        if not text:
            raise ProviderResponseError(error_message)
        return text
