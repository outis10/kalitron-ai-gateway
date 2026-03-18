from app.models.responses import Decision, OCRResult, VisionResult
from app.services.rules_engine import _parse_date, rules_engine
from app.services.scoring_service import scoring_service


def test_parse_month_year_expiry_format():
    parsed = _parse_date("2030/12")
    assert parsed is not None
    assert parsed.year == 2030
    assert parsed.month == 12
    assert parsed.day == 31


def test_unknown_expiry_is_flagged_without_failing_rule():
    result = rules_engine.validate_identity(
        OCRResult(
            raw_text="NOMBRE: JUAN PEREZ",
            structured_fields={
                "full_name": "JUAN PEREZ",
                "id_number": "ABC123",
                "expiry_date": "vigente",
            },
            confidence=0.9,
        ),
        "INE",
    )

    assert "expiry_not_past" not in result.failed_rules
    assert "unknown_expiry" in result.flags


def test_unknown_expiry_caps_decision_at_human_review():
    ocr_result = OCRResult(
        raw_text="NOMBRE: JUAN PEREZ",
        structured_fields={
            "full_name": "JUAN PEREZ",
            "id_number": "ABC123",
            "expiry_date": "vigente",
        },
        confidence=0.98,
    )
    rules_result = rules_engine.validate_identity(ocr_result, "INE")
    score = scoring_service.calculate_score(
        ocr_result,
        VisionResult(
            is_authentic=True,
            fraud_indicators=[],
            authenticity_score=0.99,
            notes="ok",
        ),
        rules_result,
    )

    assert score.decision == Decision.HUMAN_REVIEW


def test_expired_document_forces_rejection():
    ocr_result = OCRResult(
        raw_text="NOMBRE: ANA",
        structured_fields={"full_name": "ANA", "id_number": "ABC123", "expiry_date": "2018-12-31"},
        confidence=0.99,
    )
    rules_result = rules_engine.validate_identity(ocr_result, "INE")
    score = scoring_service.calculate_score(
        ocr_result,
        VisionResult(
            is_authentic=True,
            fraud_indicators=[],
            authenticity_score=0.99,
            notes="ok",
        ),
        rules_result,
    )

    assert "expired_document" in rules_result.flags
    assert score.decision == Decision.AUTO_REJECTED
