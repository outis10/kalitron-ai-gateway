from app.core.config import Settings
from app.services.anthropic_provider import AnthropicOCRService, AnthropicVisionService
from app.services.openai_provider import OpenAIOCRService, OpenAIVisionService
from app.services.provider_factory import build_ocr_service, build_vision_service


def test_provider_for_pipeline_falls_back_to_global_provider():
    config = Settings(
        AI_PROVIDER="openai",
        AI_PROVIDER_IDENTITY="",
        AI_PROVIDER_RECEIPT="",
    )

    assert config.provider_for_pipeline("identity") == "openai"
    assert config.provider_for_pipeline("receipt") == "openai"


def test_provider_factory_supports_different_providers_per_pipeline():
    config = Settings(
        AI_PROVIDER="anthropic",
        AI_PROVIDER_IDENTITY="anthropic",
        AI_PROVIDER_RECEIPT="openai",
        ANTHROPIC_API_KEY="sk-ant-test",
        OPENAI_API_KEY="sk-openai-test",
    )

    assert isinstance(build_ocr_service(config, pipeline_name="identity"), AnthropicOCRService)
    assert isinstance(
        build_vision_service(config, pipeline_name="identity"), AnthropicVisionService
    )
    assert isinstance(build_ocr_service(config, pipeline_name="receipt"), OpenAIOCRService)
    assert isinstance(build_vision_service(config, pipeline_name="receipt"), OpenAIVisionService)
