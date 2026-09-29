"""Evidence parsers, normalized evidence models, and structured-input routing."""

from .ecar import EcarDiagnostics, EcarParseResult, parse_ecar, parse_ecar_records
from .extract import (
    StructuredInputDiagnostics,
    StructuredParseResult,
    detect_structured_format,
    extract_evidence,
    parse_structured_evidence,
)
from .winlogs import (
    WinlogDiagnostics,
    WinlogParseResult,
    parse_windows_xml,
    parse_windows_xml_detailed,
    parse_winlogs,
    parse_winlogs_detailed,
)

__all__ = [
    "EcarDiagnostics",
    "EcarParseResult",
    "StructuredInputDiagnostics",
    "StructuredParseResult",
    "WinlogDiagnostics",
    "WinlogParseResult",
    "detect_structured_format",
    "extract_evidence",
    "parse_ecar",
    "parse_ecar_records",
    "parse_structured_evidence",
    "parse_windows_xml",
    "parse_windows_xml_detailed",
    "parse_winlogs",
    "parse_winlogs_detailed",
]
