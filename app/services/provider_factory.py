from app.core.config import settings
from app.core.errors import RuntimeConfigurationError
from app.services.ai_interfaces import OCRProvider, VisionProvider
from app.services.anthropic_provider import AnthropicOCRService, AnthropicVisionService
from app.services.ollama_provider import OllamaOCRService, OllamaVisionService
from app.services.openai_provider import OpenAIOCRService, OpenAIVisionService


def build_ocr_service(config=settings, pipeline_name: str = "identity") -> OCRProvider:
    provider = config.provider_for_stage(pipeline_name, "ocr")
    if provider == "anthropic":
        return AnthropicOCRService(config)
    if provider == "openai":
        return OpenAIOCRService(config)
    if provider == "ollama":
        return OllamaOCRService(config)
    raise RuntimeConfigurationError(
        f"Unsupported provider '{provider}' for pipeline '{pipeline_name}'."
    )


def build_vision_service(config=settings, pipeline_name: str = "identity") -> VisionProvider:
    provider = config.provider_for_stage(pipeline_name, "vision")
    if provider == "anthropic":
        return AnthropicVisionService(config)
    if provider == "openai":
        return OpenAIVisionService(config)
    if provider == "ollama":
        return OllamaVisionService(config)
    raise RuntimeConfigurationError(
        f"Unsupported provider '{provider}' for pipeline '{pipeline_name}'."
    )
