import pandas as pd

def save_report(results):
    df = pd.DataFrame(results)
    
    # Save to CSV format
    df.to_csv(
        "evaluation_results.csv",
        index=False,
    )
    
    # Save to Excel format
    df.to_excel(
        "evaluation_results.xlsx",
        index=False,
    )
    
    print("\nEvaluation report saved.")
    print("Files: evaluation_results.csv")
    print("       evaluation_results.xlsx")
    
    return df