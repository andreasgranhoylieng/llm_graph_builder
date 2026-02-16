"""
End-to-end benchmark utilities for grounded GraphRAG evaluation.
"""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class BenchmarkCase:
    question: str
    expected_keywords: List[str]
    must_be_grounded: bool = True
    max_confidence_if_ungrounded: float = 0.25


def _contains_all_keywords(answer: str, expected_keywords: List[str]) -> bool:
    lowered = (answer or "").lower()
    return all(keyword.lower() in lowered for keyword in expected_keywords)


def run_hybrid_benchmark(controller: Any, cases: List[BenchmarkCase]) -> Dict[str, Any]:
    """
    Execute benchmark cases against controller.chat_hybrid and score outcomes.
    """
    results: List[Dict[str, Any]] = []
    passed = 0

    for case in cases:
        response = controller.chat_hybrid(case.question)
        answer = str(response.get("answer", ""))
        confidence = float(response.get("confidence", 0) or 0)
        grounded_hits = int(response.get("doc_chunks_used", 0) or 0)

        keyword_match = _contains_all_keywords(answer, case.expected_keywords)
        grounded_ok = (grounded_hits > 0) if case.must_be_grounded else (
            confidence <= case.max_confidence_if_ungrounded
        )

        case_passed = keyword_match and grounded_ok and response.get("status") == "success"
        if case_passed:
            passed += 1

        results.append(
            {
                "question": case.question,
                "status": response.get("status"),
                "method": response.get("method"),
                "confidence": confidence,
                "doc_chunks_used": grounded_hits,
                "keyword_match": keyword_match,
                "grounded_ok": grounded_ok,
                "passed": case_passed,
                "answer_preview": answer[:240],
            }
        )

    total = len(cases)
    return {
        "total_cases": total,
        "passed_cases": passed,
        "pass_rate": round(passed / max(total, 1), 3),
        "results": results,
    }

