import os

import pytest

from src.controllers.graph_controller import GraphController
from src.evaluation.benchmark import BenchmarkCase, run_hybrid_benchmark


RUN_E2E = os.getenv("RUN_E2E_BENCHMARK", "").strip().lower() in {"1", "true", "yes"}


@pytest.mark.skipif(
    not RUN_E2E,
    reason="Set RUN_E2E_BENCHMARK=1 to run live end-to-end GraphRAG benchmark.",
)
def test_hybrid_benchmark_live():
    controller = GraphController()

    cases = [
        BenchmarkCase(
            question="How much did Google invest in SpaceX and what ownership stake did it get?",
            expected_keywords=["$1 billion", "8.33%"],
            must_be_grounded=True,
        ),
        BenchmarkCase(
            question="Who developed Claude and who founded xAI?",
            expected_keywords=["Anthropic", "Elon Musk"],
            must_be_grounded=True,
        ),
        BenchmarkCase(
            question="What is the capital of France?",
            expected_keywords=["could not find grounded evidence"],
            must_be_grounded=False,
            max_confidence_if_ungrounded=0.25,
        ),
    ]

    report = run_hybrid_benchmark(controller, cases)
    assert report["pass_rate"] >= 0.66

