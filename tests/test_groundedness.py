from executor import generate_response
from evaluator import evaluate_response


def test_refund_policy_groundedness():
    question = "Can a customer get a refund after 60 days?"

    context = (
        "Customers can request a refund within 30 days "
        "of purchase. The product must be unused, and "
        "proof of purchase is required."
    )

    expected = (
        "The policy does not allow a refund request "
        "after 60 days."
    )

    actual = generate_response(question, context)

    evaluation = evaluate_response(
        question=question,
        context=context,
        expected=expected,
        actual=actual,
    )

    assert evaluation["Faithfulness Pass"], (
        evaluation["Faithfulness Reason"]
    )
