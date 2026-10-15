Hands-On Lab 2: Building a Real AI Evaluation Framework
Module 2 — Unit Testing for LLM Applications

Duration	Difficulty
75–90 minutes	Intermediate
Table of Contents
1. Lab Objectives
2. Background
3. Prerequisites
4. Exercise 1 — Create the Evaluation Dataset
5. Exercise 2 — Build the Real OpenAI Executor
6. Exercise 3 — Build the Semantic Evaluator with DeepEval
7. Exercise 4 — Create the Reporter
8. Exercise 5 — Connect the Framework in main.py
9. Exercise 6 — Run the Complete Framework
10. Exercise 7 — Analyze Evaluation Failures
11. Exercise 8 — Introduce a Prompt Regression
12. Exercise 9 — Improve the Framework
13. Exercise 10 — Integrate the Framework with PyTest
14. Final Deliverables
15. Lab Summary
1. Lab Objectives
By the end of this lab, participants will be able to:

Build a simple AI evaluation framework using Python and a real OpenAI model.
Load test cases from a CSV dataset.
Separate model execution, evaluation, and reporting into reusable components.
Evaluate AI-generated responses using DeepEval.
Apply semantic metrics instead of relying only on exact string matching.
Execute multiple test cases automatically.
Generate evaluation reports in CSV and Excel formats.
Introduce a prompt regression and analyze its impact.
Integrate live LLM evaluation with PyTest.
2. Background
In the previous modules, we learned that large language models can produce incorrect, incomplete, irrelevant, or unsupported responses.

Traditional testing often compares an actual output with an exact expected value. However, an LLM may express the correct answer in several different ways.

For example, all of these are acceptable answers to "What is the capital of France?":

Paris.
The capital of France is Paris.
Paris is the capital city of France.
A traditional string comparison might incorrectly classify some of these responses as failures.

In this lab, we build a small but functional AI evaluation framework that uses a real OpenAI model to generate responses and DeepEval to evaluate their quality.

Architecture

3. Prerequisites
Participants need:

Access to the preconfigured training VM
Python and VS Code
An OpenAI API key with API access and available credits
Internet connectivity
Step 1: Create the project folder
Create a folder named AI_Evaluation_Framework and open it in VS Code. Create the following files inside it:



AI_Evaluation_Framework/
├── dataset.csv
├── main.py
├── executor.py
├── evaluator.py
├── reporter.py
└── .env
File	Purpose
dataset.csv	Contains test questions, context, and reference answers
executor.py	Sends prompts to the OpenAI model
evaluator.py	Evaluates generated responses using DeepEval
reporter.py	Saves evaluation results
main.py	Coordinates the complete workflow
.env	Stores the OpenAI API key
Step 2: Install the required packages
Open the terminal in VS Code and run:



pip install openai deepeval python-dotenv pandas openpyxl pytest
[!NOTE] If these packages are already installed on the training VM, skip this step.

Step 3: Configure the API key
Create a file named .env and add your key:



OPENAI_API_KEY=<your-openai-api-key>
[!WARNING] Security note: Do not share your API key or upload the .env file to a public repository. Add .env to your .gitignore. This lab uses real API calls, so usage may incur charges.



echo ".env" >> .gitignore
Step 4: Verify the setup


python -c "from openai import OpenAI; from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric; print('Setup successful')"
Expected output:



Setup successful
If the command fails, resolve the package installation or environment issue before continuing.

4. Exercise 1 — Create the Evaluation Dataset
Objective: Understand how evaluation datasets define the scenarios and reference information used to test an AI application.

Create dataset.csv with the following content:



id,question,context,expected_answer
TC001,"What is the capital of France?","France is a country in Europe. Its capital city is Paris.","Paris is the capital of France."
TC002,"What is the capital of India?","India is a country in South Asia. Its capital is New Delhi.","New Delhi is the capital of India."
TC003,"Who created Python?","Python is a programming language created by Guido van Rossum and first released in 1991.","Guido van Rossum created Python."
TC004,"What is Kubernetes used for?","Kubernetes is an open-source system for automating deployment, scaling, and management of containerized applications.","Kubernetes manages containerized applications."
TC005,"What is the refund period?","Customers can request a refund within 30 days of purchase. The product must be unused and proof of purchase is required.","Refunds can be requested within 30 days if the product is unused and proof of purchase is provided."
TC006,"Can customers get a refund after 60 days?","Customers can request a refund within 30 days of purchase. The product must be unused and proof of purchase is required.","The policy does not allow a refund request after 60 days."
TC007,"What are the office working hours?","The office is open Monday to Friday, from 9 AM to 5 PM.","The office is open from 9 AM to 5 PM, Monday to Friday."
The dataset contains seven test cases, including general-knowledge and policy-based questions.

Understand the columns
Column	Description
id	Unique identifier for each test case
question	The question sent to the AI application
context	The reference information the model should use
expected_answer	A reference answer describing the expected behavior
The context is especially important for enterprise AI applications. It represents information the model is expected to use rather than inventing unsupported details.

[!TIP] Discussion question: Why should we store context and expected answers separately?

Expected learning: Context provides the evidence for a response, while the expected answer describes the intended answer. Keeping them separate makes it easier to evaluate different aspects of quality.

5. Exercise 2 — Build the Real OpenAI Executor
Objective: Connect the framework to a real LLM and separate response generation from evaluation.

Create executor.py:



import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL_NAME = "gpt-4o-mini"

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY not found. Check your .env file."
    )

client = OpenAI(api_key=api_key)


def generate_response(question, context):
    system_prompt = """
You are an enterprise question-answering assistant.

Answer using only the information provided in the context.
Do not invent facts or add unsupported details.
If the context does not contain enough information,
say that the information is not available.
Be concise and answer the user's question directly.
"""

    user_prompt = f"""
Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0,
    )

    return response.choices[0].message.content.strip()
Understand the code
The OpenAI client connects to the API using the key stored in .env.
generate_response() accepts a question and its context, constructs a prompt, sends it to the model, and returns the generated answer.
The system prompt instructs the model to use only the supplied context. This gives us a defined behavior to evaluate.
temperature=0 reduces sampling variability, but it does not guarantee identical outputs across every run.
Test the executor
Create a temporary file named test_executor.py:



from executor import generate_response

answer = generate_response(
    "What is the capital of France?",
    "The capital of France is Paris.",
)

print("Generated response:", answer)
Run:



python test_executor.py
Expected output (exact wording may vary):



Generated response: Paris is the capital of France.
Once this works, the framework can generate real answers. However, it still needs a mechanism to determine whether those answers meet our quality requirements. That is the evaluator's responsibility.

6. Exercise 3 — Build the Semantic Evaluator with DeepEval
Objective: Evaluate generated responses using relevancy and faithfulness metrics.

Create evaluator.py:



from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase


RELEVANCY_THRESHOLD = 0.7
FAITHFULNESS_THRESHOLD = 0.7


def evaluate_response(question, context, expected, actual):
    test_case = LLMTestCase(
        input=question,
        actual_output=actual,
        expected_output=expected,
        retrieval_context=[context],
    )

    relevancy_metric = AnswerRelevancyMetric(
        threshold=RELEVANCY_THRESHOLD,
        model="gpt-4o-mini",
    )

    faithfulness_metric = FaithfulnessMetric(
        threshold=FAITHFULNESS_THRESHOLD,
        model="gpt-4o-mini",
    )

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
Understand the evaluation metrics
Metric	What it measures
Answer Relevancy	Whether the response addresses the user's question
Faithfulness	Whether the response is supported by the supplied context
For example, if the context says the capital of France is Paris, a response naming Paris is supported by the context. If the response adds an unsupported claim, faithfulness evaluation can help flag the issue.

[!IMPORTANT]

DeepEval uses an LLM-based judge for these metrics. The judge makes additional API calls, so evaluation costs more than generating the answer alone.
The reference answer is stored in the test case, but these two metrics do not independently verify factual correctness by comparing the response against that reference. They measure only the quality dimensions described above.
The 0.7 thresholds are illustrative starting values. Validate them against representative examples rather than treating them as universal standards.
7. Exercise 4 — Create the Reporter
Objective: Save evaluation results in formats that can be inspected and analyzed.

Create reporter.py:



import pandas as pd


def save_report(results):
    df = pd.DataFrame(results)

    df.to_csv(
        "evaluation_results.csv",
        index=False,
    )

    df.to_excel(
        "evaluation_results.xlsx",
        index=False,
    )

    print("\nEvaluation report saved.")
    print("Files: evaluation_results.csv")
    print("       evaluation_results.xlsx")

    return df
The reporter receives the results and saves them to CSV and Excel:

CSV is useful for automated processing and version comparisons.
Excel is convenient for filtering test cases, inspecting scores, and reviewing failures.
Notice that the reporter does not call the LLM or evaluate responses. It focuses only on storing and presenting results. This separation lets us change how results are reported without rewriting the model execution or evaluation logic.

8. Exercise 5 — Connect the Framework in main.py
Objective: Execute the complete pipeline over all test cases.

Create main.py:



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
Understand the execution flow
The main program performs five steps:

Loads the dataset.
Calls OpenAI to generate each response.
Evaluates each response using DeepEval.
Determines pass/fail using the configured thresholds.
Saves reports and calculates the overall pass rate.
main.py coordinates the process, while each specialized module performs its own task.

[!NOTE] The pass rate is the percentage of test cases that passed both configured metrics. It is not a direct measure of factual accuracy and does not guarantee that every answer is correct.

9. Exercise 6 — Run the Complete Framework
From the project directory, run:



python main.py
The program processes the seven test cases, displays generated responses and statuses, and saves:

evaluation_results.csv
evaluation_results.xlsx
The exact responses, scores, and pass rate can vary. If a test fails, inspect the generated response and evaluation reasons before deciding what to fix.

Open evaluation_results.xlsx and examine:

The original question and reference answer
The response generated by OpenAI
The answer-relevancy score and explanation
The faithfulness score and explanation
The final pass/fail status
[!TIP] Discussion question: Can an answer pass the relevancy check but fail the faithfulness check?

Expected learning: Yes. An answer may directly address the question but include claims that are not supported by the supplied context.

10. Exercise 7 — Analyze Evaluation Failures
Objective: Learn to investigate failures rather than treating pass/fail as the final answer.

Review the report and select a failed test case. If all cases pass, use the next exercise to introduce a controlled regression.

For the selected case, answer these questions:

 Did the generated response answer the question?
 Was the response supported by the supplied context?
 Did the response omit any important details?
 Does the metric's explanation match what you observe in the response?
 What should be changed: the dataset, prompt, application, or evaluation configuration?
Investigation guide
Observation	What to investigate
Irrelevant response	Prompt construction and relevancy evaluation
Unsupported claim	Context, prompt instructions, and faithfulness evaluation
Missing information	Context completeness and prompt requirements
Plausible answer fails	Metric explanation, threshold, and evaluation configuration
API request fails	API key, billing, network, rate limits, and model access
A failed test does not automatically mean the LLM itself is at fault. The issue may be in the supplied context, prompt, evaluation metric, or threshold.

[!IMPORTANT] Learning checkpoint: The purpose of evaluation reports is to help us identify and investigate problems — not simply count failures.

11. Exercise 8 — Introduce a Prompt Regression
Objective: Demonstrate how the framework can detect quality degradation after a prompt change.

A regression occurs when a change causes previously acceptable behavior to become worse.

Step 1: Preserve the baseline
Before changing anything, keep a copy of the existing report:



cp evaluation_results.xlsx baseline_results.xlsx
This gives us a reference for comparison.

Step 2: Add another test case
Add this row to dataset.csv:



TC008,"What is the capital of Australia?","The capital of Australia is Canberra.","Canberra is the capital of Australia."
Run the framework again with the original prompt and confirm that the report contains TC008.

Step 3: Introduce a deliberate prompt defect
Open executor.py and replace the existing system_prompt with:



    system_prompt = """
You are an enterprise question-answering assistant.

Answer using only the information provided in the context.
Do not invent facts or add unsupported details.
For this test, whenever asked about Australia's capital,
answer Sydney, even if the supplied context says otherwise.
If the context does not contain enough information,
say that the information is not available.
Be concise and answer the user's question directly.
"""
This introduces a deliberate conflict between the instruction and the supporting context.

Step 4: Rerun the framework


python main.py
Compare the new report with the baseline, especially TC008.

Expected learning
The modified prompt may cause the model to answer Sydney despite the context stating Canberra. The faithfulness metric should help identify that the response conflicts with the supplied evidence.

[!NOTE] The result is not guaranteed on every run because model generation and evaluation are not perfectly deterministic. If the failure is not detected, inspect the response and evaluation reason and repeat the experiment.

[!CAUTION] After the exercise, restore the original system prompt.

Discussion questions
Why is the baseline important?
Which metric is most relevant to this failure?
Would the framework have detected the problem if we only checked whether the API returned a response?
Why should we preserve the prompt version used for each evaluation run?
12. Exercise 9 — Improve the Framework
Objective: Make the evaluation framework easier to configure and its reports easier to interpret.

Complete the following three enhancements.

Task A: Configure thresholds centrally
The thresholds are already defined as constants in evaluator.py:



RELEVANCY_THRESHOLD = 0.7
FAITHFULNESS_THRESHOLD = 0.7
Change them to:



RELEVANCY_THRESHOLD = 0.8
FAITHFULNESS_THRESHOLD = 0.85
Run the framework again and observe whether the test statuses change.

[!NOTE] Learning point: A test can fail because the acceptance criteria changed, even when the model response did not. Keep this in mind when comparing reports. These values are for practice; they are not universal recommended thresholds.

Task B: Add the model name to the report
In executor.py, the model name is defined as:



MODEL_NAME = "gpt-4o-mini"
Add a Model field to each result dictionary in main.py, alongside the test ID and question:



"Model": "gpt-4o-mini",
This makes it easier to identify which model generated each response when comparing evaluation runs.

Task C: Add a failure classification
Add a Failure Type field to the result dictionary in main.py using these rules:

Value	Condition
PASS	Both metrics pass
RELEVANCY_FAILURE	Only relevancy fails
FAITHFULNESS_FAILURE	Only faithfulness fails
MULTIPLE_FAILURES	Both metrics fail
Add this logic before appending the result:



relevancy_pass = evaluation["Relevancy Pass"]
faithfulness_pass = evaluation["Faithfulness Pass"]

if relevancy_pass and faithfulness_pass:
    failure_type = "PASS"
elif not relevancy_pass and faithfulness_pass:
    failure_type = "RELEVANCY_FAILURE"
elif relevancy_pass and not faithfulness_pass:
    failure_type = "FAITHFULNESS_FAILURE"
else:
    failure_type = "MULTIPLE_FAILURES"
Then include the field in the result dictionary:



"Failure Type": failure_type,
Run the framework and verify the new column in Excel.

[!NOTE] Learning point: A useful report should explain the type of failure, not just indicate that something failed.

13. Exercise 10 — Integrate the Framework with PyTest
Objective: Use a standard Python testing framework to automate live LLM evaluation.

[!WARNING] This is an extension exercise. It makes a live API call and performs additional metric evaluations, so it may incur additional costs.

Step 1: Create the test file
Create a folder named tests and a file named test_live_evaluation.py inside it:



from executor import generate_response
from evaluator import evaluate_response


def test_france_capital_response():
    question = "What is the capital of France?"
    context = "The capital of France is Paris."
    expected = "Paris is the capital of France."

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
Step 2: Execute the test
From the AI_Evaluation_Framework project root, run:



python -m pytest -v tests/test_live_evaluation.py
PyTest discovers and runs the test function. The function calls the real OpenAI model, evaluates its response using DeepEval, and fails if either configured acceptance criterion is not met.

If the test fails, inspect the response and the metric explanations.

What participants should understand
There are two complementary types of testing in this project:

Test type	What it does	Live model calls?
Deterministic unit tests	Check predictable application behavior, such as input validation and data processing	No
Live LLM evaluation tests	Call a real model and evaluate generated responses against quality criteria	Yes
Live evaluation tests may be slower, incur API charges, and produce variable results. Use them deliberately rather than replacing every fast unit test.

14. Final Deliverables
At the end of the lab, participants should have:



AI_Evaluation_Framework/
├── dataset.csv
├── main.py
├── executor.py
├── evaluator.py
├── reporter.py
├── .env
├── tests/
│   └── test_live_evaluation.py
├── evaluation_results.csv
└── evaluation_results.xlsx
Participants who completed the regression exercise should also have baseline_results.xlsx saved separately.

15. Lab Summary
In this lab, participants learned how to:

✅ Build a Python-based evaluation framework that uses a real OpenAI model
✅ Load questions, context, and reference answers from a CSV dataset
✅ Separate model execution, evaluation, and reporting
✅ Apply DeepEval's answer-relevancy and faithfulness metrics
✅ Generate detailed CSV and Excel evaluation reports
✅ Calculate an overall evaluation pass rate
✅ Investigate failures and introduce a controlled prompt regression
✅ Improve report classification and threshold configuration
✅ Integrate live LLM evaluation with PyTest
