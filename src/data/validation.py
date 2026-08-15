"""Label and schema validation for Threat Matrix records."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.data.constants import (
    BINARY_BENIGN,
    BINARY_MALICIOUS,
    INTENT_LABELS,
    INTENT_TO_LABEL,
    MAX_SEVERITY,
    MIN_SEVERITY,
    REQUIRED_FIELDS,
)


@dataclass
class ValidationIssue:
    row_index: int
    field: str
    message: str
    severity: str = "error"


@dataclass
class ValidationReport:
    total_rows: int = 0
    valid_rows: int = 0
    invalid_rows: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_rows": self.total_rows,
            "valid_rows": self.valid_rows,
            "invalid_rows": self.invalid_rows,
            "issue_count": len(self.issues),
            "issues": [
                {
                    "row_index": issue.row_index,
                    "field": issue.field,
                    "message": issue.message,
                    "severity": issue.severity,
                }
                for issue in self.issues[:100]
            ],
        }


def _as_int(value: object, default: int | None = None) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def validate_record(record: dict, row_index: int) -> list[ValidationIssue]:
    """Validate one dataset record against Threat Matrix schema rules."""
    issues: list[ValidationIssue] = []

    for field_name in REQUIRED_FIELDS:
        if field_name not in record or record[field_name] in (None, ""):
            if field_name == "technique" and record.get("intent_label") == 0:
                continue
            issues.append(
                ValidationIssue(
                    row_index=row_index,
                    field=field_name,
                    message=f"Missing required field: {field_name}",
                )
            )

    text = record.get("text_clean", record.get("text", ""))
    if not str(text).strip():
        issues.append(
            ValidationIssue(
                row_index=row_index,
                field="text",
                message="Empty text after cleaning",
            )
        )

    intent_label = _as_int(record.get("intent_label", record.get("label")))
    binary_label = _as_int(record.get("binary_label"))
    severity = _as_int(record.get("severity"))
    intent_name = str(record.get("intent", "")).strip()
    technique = str(record.get("technique", "")).strip()

    if intent_label is None or intent_label not in INTENT_LABELS:
        issues.append(
            ValidationIssue(
                row_index=row_index,
                field="intent_label",
                message=f"Invalid intent_label: {record.get('intent_label')}",
            )
        )
    elif intent_name and intent_name != INTENT_LABELS[intent_label]:
        issues.append(
            ValidationIssue(
                row_index=row_index,
                field="intent",
                message=(
                    f"Intent name '{intent_name}' does not match "
                    f"intent_label {intent_label} ({INTENT_LABELS[intent_label]})"
                ),
            )
        )
    elif intent_name and intent_name in INTENT_TO_LABEL:
        if INTENT_TO_LABEL[intent_name] != intent_label:
            issues.append(
                ValidationIssue(
                    row_index=row_index,
                    field="intent",
                    message="Intent string and intent_label are inconsistent",
                )
            )

    if binary_label not in (BINARY_BENIGN, BINARY_MALICIOUS):
        issues.append(
            ValidationIssue(
                row_index=row_index,
                field="binary_label",
                message=f"Invalid binary_label: {record.get('binary_label')}",
            )
        )
    elif intent_label is not None:
        expected_binary = BINARY_BENIGN if intent_label == 0 else BINARY_MALICIOUS
        if binary_label != expected_binary:
            issues.append(
                ValidationIssue(
                    row_index=row_index,
                    field="binary_label",
                    message=(
                        f"binary_label={binary_label} inconsistent with "
                        f"intent_label={intent_label}"
                    ),
                )
            )

    if severity is None or not (MIN_SEVERITY <= severity <= MAX_SEVERITY):
        issues.append(
            ValidationIssue(
                row_index=row_index,
                field="severity",
                message=f"Invalid severity: {record.get('severity')}",
            )
        )
    elif intent_label == 0 and severity > 2:
        issues.append(
            ValidationIssue(
                row_index=row_index,
                field="severity",
                message=f"Benign sample has unusually high severity: {severity}",
                severity="warning",
            )
        )

    if intent_label not in (None, 0) and not technique:
        issues.append(
            ValidationIssue(
                row_index=row_index,
                field="technique",
                message="Malicious sample missing technique label",
            )
        )

    return issues


def validate_records(records: list[dict]) -> ValidationReport:
    """Validate a list of records and summarize results."""
    report = ValidationReport(total_rows=len(records))
    invalid_indices: set[int] = set()

    for idx, record in enumerate(records):
        row_issues = validate_record(record, idx)
        if row_issues:
            error_issues = [i for i in row_issues if i.severity == "error"]
            if error_issues:
                invalid_indices.add(idx)
            report.issues.extend(row_issues)

    report.invalid_rows = len(invalid_indices)
    report.valid_rows = report.total_rows - report.invalid_rows
    return report


def filter_valid_records(records: list[dict]) -> tuple[list[dict], ValidationReport]:
    """Return only rows without validation errors."""
    report = validate_records(records)
    invalid = {
        issue.row_index
        for issue in report.issues
        if issue.severity == "error"
    }
    valid_records = [row for idx, row in enumerate(records) if idx not in invalid]
    return valid_records, report
