from app.core.config import settings
from app.services.provider_factory import build_vision_service

identity_vision_service = build_vision_service(settings, pipeline_name="identity")
receipt_vision_service = build_vision_service(settings, pipeline_name="receipt")
