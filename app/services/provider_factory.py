from app.core.config import settings
from app.core.errors import RuntimeConfigurationError
from app.services.ai_interfaces import OCRProvider, VisionProvider
from app.services.anthropic_provider import AnthropicOCRService, AnthropicVisionService
from app.services.openai_provider import OpenAIOCRService, OpenAIVisionService


def build_ocr_service(config=settings, pipeline_name: str = "identity") -> OCRProvider:
    provider = config.provider_for_pipeline(pipeline_name)
    if provider == "anthropic":
        return AnthropicOCRService(config)
    if provider == "openai":
        return OpenAIOCRService(config)
    raise RuntimeConfigurationError(
        f"Unsupported provider '{provider}' for pipeline '{pipeline_name}'."
    )


def build_vision_service(config=settings, pipeline_name: str = "identity") -> VisionProvider:
    provider = config.provider_for_pipeline(pipeline_name)
    if provider == "anthropic":
        return AnthropicVisionService(config)
    if provider == "openai":
        return OpenAIVisionService(config)
    raise RuntimeConfigurationError(
        f"Unsupported provider '{provider}' for pipeline '{pipeline_name}'."
    )
