import pytest

from executor import generate_response
from evaluator import evaluate_response


@pytest.mark.parametrize(
    "question,context,expected",
    [
        (
            "What is the capital of France?",
            "The capital of France is Paris.",
            "Paris is the capital of France.",
        ),
        (
            "What is the capital of India?",
            "The capital of India is New Delhi.",
            "New Delhi is the capital of India.",
        ),
        (
            "Who created Python?",
            "Python was created by Guido van Rossum.",
            "Guido van Rossum created Python.",
        ),
        (
            "What is the capital of Japan?",
            "The capital of Japan is Tokyo.",
            "Tokyo is the capital of Japan.",
        ),
        (
            "What is Kubernetes used for?",
            "Kubernetes automates deployment, scaling, and management of containerized applications.",
            "Kubernetes manages containerized applications.",
        ),

    ],
)
def test_answer_quality(question, context, expected):
    actual = generate_response(question, context)

    evaluation = evaluate_response(
        question=question,
        context=context,
        expected=expected,
        actual=actual,
    )

    assert evaluation["Relevancy Pass"], (
        evaluation["Relevancy Reason"]
    )

    assert evaluation["Faithfulness Pass"], (
        evaluation["Faithfulness Reason"]
    )
