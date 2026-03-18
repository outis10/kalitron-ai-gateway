import logging
import time
from abc import ABC, abstractmethod
from uuid import UUID

from app.models.responses import BaseValidationResponse

logger = logging.getLogger(__name__)


class BasePipeline(ABC):
    """
    Abstract base class for all document validation pipelines.

    Subclasses must implement `process()`, which orchestrates the OCR →
    Vision → Rules → Scoring steps and returns a typed response.
    """

    @abstractmethod
    async def process(
        self,
        image_bytes: bytes,
        media_type: str,
        metadata: dict,
    ) -> BaseValidationResponse:
        """
        Run the full validation pipeline for a document image.

        Args:
            image_bytes: Raw bytes of the uploaded document image.
            media_type:  MIME type of the image (e.g. 'image/jpeg').
            metadata:    Contextual data (client_id, source, document_type, etc.).

        Returns:
            A subclass of BaseValidationResponse with all validation results.
        """

    def _start_timer(self) -> float:
        """Return the current monotonic time in seconds."""
        return time.monotonic()

    def _elapsed_ms(self, start: float) -> float:
        """Return elapsed milliseconds since `start`."""
        return round((time.monotonic() - start) * 1000, 2)

    def _log_result(
        self,
        request_id: UUID,
        document_type: str,
        final_score: float,
        decision: str,
        elapsed_ms: float,
    ) -> None:
        """Emit a structured INFO log with key pipeline outcome metrics."""
        logger.info(
            "Pipeline completed | request_id=%s document_type=%s final_score=%.2f decision=%s elapsed_ms=%.1f",
            request_id,
            document_type,
            final_score,
            decision,
            elapsed_ms,
        )
