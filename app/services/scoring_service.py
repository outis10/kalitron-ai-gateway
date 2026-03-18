import logging

from app.core.config import settings
from app.models.responses import Decision, OCRResult, RulesResult, ScoringResult, VisionResult

logger = logging.getLogger(__name__)

# Weighted contributions to the final score (must sum to 1.0)
_WEIGHT_OCR = 0.25
_WEIGHT_VISION = 0.40
_WEIGHT_RULES = 0.35


class ScoringService:
    """Aggregates OCR, vision, and rules results into a final confidence score and routing decision."""

    def calculate_score(
        self,
        ocr_result: OCRResult,
        vision_result: VisionResult,
        rules_result: RulesResult,
    ) -> ScoringResult:
        """
        Compute a weighted confidence score and determine the routing decision.

        Weights:
            - OCR confidence:          25%
            - Vision authenticity:     40%
            - Rules compliance:        35%

        Thresholds:
            - score > SCORE_AUTO_APPROVE  → AUTO_APPROVED
            - score >= SCORE_HUMAN_REVIEW → HUMAN_REVIEW
            - score < SCORE_HUMAN_REVIEW  → AUTO_REJECTED

        Args:
            ocr_result:    Result from OCRService.
            vision_result: Result from VisionService.
            rules_result:  Result from RulesEngine.

        Returns:
            ScoringResult with final_score (0-100), decision, breakdown, and requires_human_review flag.
        """
        ocr_score_100 = ocr_result.confidence * 100.0
        vision_score_100 = vision_result.authenticity_score * 100.0
        rules_score_100 = rules_result.rules_score * 100.0

        final_score = (
            ocr_score_100 * _WEIGHT_OCR
            + vision_score_100 * _WEIGHT_VISION
            + rules_score_100 * _WEIGHT_RULES
        )
        final_score = round(final_score, 2)

        decision = self._apply_thresholds(final_score)
        if "expired_document" in rules_result.flags:
            decision = Decision.AUTO_REJECTED
        elif "unknown_expiry" in rules_result.flags and decision == Decision.AUTO_APPROVED:
            decision = Decision.HUMAN_REVIEW
        requires_human_review = decision == Decision.HUMAN_REVIEW

        breakdown = {
            "ocr_confidence": round(ocr_result.confidence, 4),
            "ocr_contribution": round(ocr_score_100 * _WEIGHT_OCR, 2),
            "vision_authenticity": round(vision_result.authenticity_score, 4),
            "vision_contribution": round(vision_score_100 * _WEIGHT_VISION, 2),
            "rules_score": round(rules_result.rules_score, 4),
            "rules_contribution": round(rules_score_100 * _WEIGHT_RULES, 2),
            "weights": {
                "ocr": _WEIGHT_OCR,
                "vision": _WEIGHT_VISION,
                "rules": _WEIGHT_RULES,
            },
            "thresholds": {
                "auto_approve": settings.SCORE_AUTO_APPROVE,
                "human_review": settings.SCORE_HUMAN_REVIEW,
            },
        }
        if rules_result.flags:
            breakdown["policy_flags"] = list(rules_result.flags)

        logger.info(
            "Score calculated: final=%.2f decision=%s (ocr=%.1f vision=%.1f rules=%.1f)",
            final_score,
            decision.value,
            ocr_score_100,
            vision_score_100,
            rules_score_100,
        )

        return ScoringResult(
            final_score=final_score,
            decision=decision,
            breakdown=breakdown,
            requires_human_review=requires_human_review,
        )

    @staticmethod
    def _apply_thresholds(score: float) -> Decision:
        """Map a numeric score to a routing decision using configured thresholds."""
        if score > settings.SCORE_AUTO_APPROVE:
            return Decision.AUTO_APPROVED
        if score >= settings.SCORE_HUMAN_REVIEW:
            return Decision.HUMAN_REVIEW
        return Decision.AUTO_REJECTED


scoring_service = ScoringService()
