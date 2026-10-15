import argparse
import json
from pathlib import Path

from executor import execute_case
from g_eval import score_response
from reporter import write_report

TIERS = {
    "smoke": {"smoke"},
    "critical": {"smoke", "critical"},
    "deep": {"smoke", "critical", "deep"}
}


def load_cases():
    path = Path("data/test_cases.jsonl")
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tier", choices=TIERS.keys(), default="smoke"
    )
    parser.add_argument("--threshold", type=float, default=4.0)
    args = parser.parse_args()

    selected = [
        case for case in load_cases()
        if case.get("tier") in TIERS[args.tier]
    ]

    if not selected:
        raise RuntimeError("No test cases selected.")

    results = []

    for index, case in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] {case['id']}")

        try:
            answer = execute_case(case)
            score = score_response(case, answer)

            results.append({
                "id": case["id"],
                "question": case["question"],
                "answer": answer,
                **score
            })

        except Exception as exc:
            results.append({
                "id": case["id"],
                "question": case["question"],
                "answer": "",
                "correctness": 0,
                "relevance": 0,
                "faithfulness": 0,
                "safety": 0,
                "average_score": 0,
                "reason": f"Execution/evaluation error: {exc}"
            })

    summary = write_report(results, args.tier, args.threshold)

    print(json.dumps({
        key: summary[key]
        for key in [
            "tier", "total", "passed", "failed",
            "average_score", "gate_passed"
        ]
    }, indent=2))

    if not summary["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
