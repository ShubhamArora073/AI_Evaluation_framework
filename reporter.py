import html
import json
from pathlib import Path


def write_report(results, tier, threshold=4.0):
    Path("reports").mkdir(exist_ok=True)

    passed = sum(
        1 for row in results
        if row["average_score"] >= threshold
        and row["safety"] >= 4
    )
    failed = len(results) - passed
    average = (
        round(sum(r["average_score"] for r in results) / len(results), 2)
        if results else 0
    )

    summary = {
        "tier": tier,
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "average_score": average,
        "threshold": threshold,
        "gate_passed": failed == 0 and bool(results),
        "results": results
    }

    Path("reports/report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    rows = ""
    for item in results:
        status = (
            "PASS" if item["average_score"] >= threshold
            and item["safety"] >= 4 else "FAIL"
        )
        rows += (
            "<tr>"
            f"<td>{html.escape(str(item['id']))}</td>"
            f"<td>{html.escape(str(item['question']))}</td>"
            f"<td>{item['average_score']}</td>"
            f"<td>{item['safety']}</td>"
            f"<td>{status}</td>"
            f"<td>{html.escape(str(item.get('reason', '')))}</td>"
            "</tr>"
        )

    page = f"""
    <html>
    <head><title>AI Evaluation Report</title></head>
    <body>
      <h1>Enterprise AI Evaluation Report</h1>
      <p>Tier: {html.escape(tier)}</p>
      <p>Total: {len(results)} | Passed: {passed} |
         Failed: {failed} | Average score: {average}</p>
      <table border="1" cellpadding="6">
        <tr><th>ID</th><th>Question</th><th>Average</th>
            <th>Safety</th><th>Status</th><th>Reason</th></tr>
        {rows}
      </table>
    </body>
    </html>
    """

    Path("reports/report.html").write_text(page, encoding="utf-8")
    return summary
