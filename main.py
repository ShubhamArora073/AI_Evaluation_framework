import pandas as pd
from executor import generate_response
from evaluator import evaluate_response
from reporter import save_report

def main():
    dataset = pd.read_csv("dataset.csv")
    results = []
    
    for _, row in dataset.iterrows():
        question = row["question"]
        context = row["context"]
        expected = row["expected_answer"]
        
        print(f"\nEvaluating: {question}")
        
        # Step 1: Generate a real response
        actual = generate_response(question, context)
        
        # Step 2: Evaluate the response
        evaluation = evaluate_response(
            question=question,
            context=context,
            expected=expected,
            actual=actual,
        )
        
        # Step 3: Determine the test status
        passed = (
            evaluation["Relevancy Pass"]
            and evaluation["Faithfulness Pass"]
        )
        
        results.append({
            "Test ID": row["id"],
            "Question": question,
            "Context": context,
            "Expected Answer": expected,
            "Generated Answer": actual,
            **evaluation,
            "Status": "PASS" if passed else "FAIL",
        })
        
        print(f"Generated: {actual}")
        print(f"Status: {'PASS' if passed else 'FAIL'}")
        
    # Step 4: Generate reports
    report = save_report(results)
    
    # Step 5: Summarize the evaluation run
    total = len(report)
    passed_count = (report["Status"] == "PASS").sum()
    failed_count = total - passed_count
    pass_rate = (
        passed_count / total * 100
        if total
        else 0
    )
    
    print("\n===== Evaluation Summary =====")
    print(f"Total test cases: {total}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print(f"Evaluation pass rate: {pass_rate:.1f}%")

if __name__ == "__main__":
    main()