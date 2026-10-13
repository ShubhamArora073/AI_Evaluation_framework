import pytest

from executor import generate_response
from evaluator import evaluate_response


@pytest.mark.parametrize(
    "question,context,expected",
    [
        (
            "What is the capital of Australia?",
            "The capital of Australia is Canberra.",
            "Canberra is the capital of Australia.",
        ),
        (
            "What is the capital of Germany?",
            "The capital of Germany is Berlin.",
            "Berlin is the capital of Germany.",
        ),
        (
            "What is the capital of Canada?",
            "The capital of Canada is Ottawa.",
            "Ottawa is the capital of Canada.",
        ),
    ],
)
def test_factual_answers(question, context, expected):
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
