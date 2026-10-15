from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

# Threshold Configurations
RELEVANCY_THRESHOLD = 0.7
FAITHFULNESS_THRESHOLD = 0.7

def evaluate_response(question, context, expected, actual):
    test_case = LLMTestCase(
        input=question,
        actual_output=actual,
        expected_output=expected,
        retrieval_context=[context],
    )
    
    # Initialize Metrics
    relevancy_metric = AnswerRelevancyMetric(
        threshold=RELEVANCY_THRESHOLD,
        model="gpt-4o-mini",
    )
    faithfulness_metric = FaithfulnessMetric(
        threshold=FAITHFULNESS_THRESHOLD,
        model="gpt-4o-mini",
    )
    
    # Run Measurements
    relevancy_metric.measure(test_case)
    faithfulness_metric.measure(test_case)
    
    relevancy_score = float(relevancy_metric.score or 0)
    faithfulness_score = float(faithfulness_metric.score or 0)
    
    return {
        "Relevancy Score": relevancy_score,
        "Relevancy Pass": (
            relevancy_score >= RELEVANCY_THRESHOLD
        ),
        "Relevancy Reason": relevancy_metric.reason or "",
        "Faithfulness Score": faithfulness_score,
        "Faithfulness Pass": (
            faithfulness_score >= FAITHFULNESS_THRESHOLD
        ),
        "Faithfulness Reason": faithfulness_metric.reason or "",
    }
