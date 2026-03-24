from app.core.config import Settings
from app.services.anthropic_provider import AnthropicOCRService, AnthropicVisionService
from app.services.ollama_provider import OllamaOCRService, OllamaVisionService
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
    assert config.provider_for_stage("identity", "ocr") == "openai"
    assert config.provider_for_stage("identity", "vision") == "openai"


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


def test_provider_factory_supports_ollama_for_vision():
    config = Settings(
        AI_PROVIDER="anthropic",
        OCR_PROVIDER_IDENTITY="openai",
        VISION_PROVIDER_IDENTITY="ollama",
        ANTHROPIC_API_KEY="sk-ant-test",
        OPENAI_API_KEY="sk-openai-test",
    )

    assert isinstance(build_ocr_service(config, pipeline_name="identity"), OpenAIOCRService)
    assert isinstance(build_vision_service(config, pipeline_name="identity"), OllamaVisionService)


def test_stage_override_takes_precedence_over_pipeline_override():
    config = Settings(
        AI_PROVIDER="anthropic",
        AI_PROVIDER_IDENTITY="openai",
        OCR_PROVIDER_IDENTITY="anthropic",
        VISION_PROVIDER_IDENTITY="ollama",
        ANTHROPIC_API_KEY="sk-ant-test",
        OPENAI_API_KEY="sk-openai-test",
    )

    assert config.provider_for_stage("identity", "ocr") == "anthropic"
    assert config.provider_for_stage("identity", "vision") == "ollama"


def test_provider_factory_supports_ollama_for_ocr():
    config = Settings(
        AI_PROVIDER="anthropic",
        OCR_PROVIDER_IDENTITY="ollama",
        VISION_PROVIDER_IDENTITY="ollama",
        ANTHROPIC_API_KEY="sk-ant-test",
    )

    assert isinstance(build_ocr_service(config, pipeline_name="identity"), OllamaOCRService)
    assert isinstance(build_vision_service(config, pipeline_name="identity"), OllamaVisionService)
