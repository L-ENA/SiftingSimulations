import os
import json
import pandas as pd
import dotenv
from sklearn.metrics import classification_report

# Reuse helper functions from batch_api_runner
from batch_api_runner import (
    load_batch_metadata,
    retrieve_batch_results_by_id,
    process_batch_results,
    calculate_batch_cost,
)


def main():
    dotenv.load_dotenv()

    # Ask user for the batch ID they want to resume
    batch_id = input("Enter the OpenAI batch ID to resume (e.g. batch_xxxxx): ").strip()
    if not batch_id:
        print("No batch ID provided. Exiting.")
        return

    # Load metadata for this batch
    metadata = load_batch_metadata(batch_id)
    if not metadata:
        print(f"No metadata found for batch {batch_id} in batch_metadata.json")
        return

    dataset_name = metadata["dataset_name"]
    managed_run_info = metadata.get("managed_run_info", {})
    model = managed_run_info.get("model", "gpt-5-mini")
    label_col = managed_run_info.get("label_col")

    if not label_col:
        print("label_col not found in managed_run_info; cannot compute classification report.")

    print("\nLoaded batch metadata:")
    print(f"  Batch ID: {batch_id}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Model: {model}")

    # Load managed_runs.json to find the original data path
    try:
        with open("managed_runs.json", "r") as f:
            managed_runs = json.load(f)
    except FileNotFoundError:
        print("managed_runs.json not found. Cannot locate original dataset path.")
        return

    if dataset_name not in managed_runs:
        print(f"Dataset {dataset_name} not found in managed_runs.json")
        return

    filename = managed_runs[dataset_name]["path"]
    outfolder = r"data\\LLM_predictions"

    # Rebuild the same dataframe ordering used in batch_api_runner
    try:
        df = pd.read_csv(filename, encoding="utf-8").fillna("")
        enc = "utf-8"
    except Exception:
        df = pd.read_csv(filename, encoding="windows-1252").fillna("")
        enc = "windows-1252"

    print(f"\nOriginal dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # The original script used seed=0 and shuffled before creating requests
    seed = 0
    df = df.sample(frac=1, random_state=seed)
    df.reset_index(drop=True, inplace=True)

    # Retrieve results for this batch ID
    print("\nRetrieving batch results from OpenAI...")
    batch_results, batch_info, _ = retrieve_batch_results_by_id(batch_id)

    if not batch_results:
        status = getattr(batch_info, "status", "unknown") if batch_info is not None else "unknown"
        print(f"No results returned for batch {batch_id}. Current status: {status}")
        print("If status is not 'completed', wait and try again later.")
        return

    # Process results into predictions/justifications/raw responses
    print("\nProcessing results...")
    predictions, justifications, raw_responses = process_batch_results(df, batch_results)

    # Attach to dataframe
    df["LLM alone"] = predictions
    df["LLM Justification"] = justifications
    df["LLM Raw Response"] = raw_responses

    # Save outputs (same pattern as batch_api_runner)
    print("\nSaving outputs...")
    os.makedirs(outfolder, exist_ok=True)
    df.to_csv(os.path.join(outfolder, filename.split("\\")[-1]), index=False, encoding=enc)
    df.to_csv("data/batch_backup_resume.csv", index=False, encoding=enc)

    # Save raw responses separately
    raw_df = pd.DataFrame({
        "index": range(len(raw_responses)),
        "raw_response": raw_responses,
    })
    raw_df.to_csv(f"data/batch_raw_responses_{dataset_name}_resume.csv", index=False, encoding=enc)

    # Calculate and display cost information
    if batch_info is not None:
        print("\n" + "=" * 60)
        print("BATCH COST ANALYSIS (RESUMED)")
        print("=" * 60)
        print(f"Model: {model}")
        cost_info = calculate_batch_cost(batch_info, model=model)
        print(f"Completed requests: {cost_info['completed_requests']}")
        print(f"Failed requests: {cost_info['failed_requests']}")
        print(f"Estimated input tokens: {cost_info['estimated_input_tokens']}")
        print(f"Estimated output tokens: {cost_info['estimated_output_tokens']}")
        print(f"Input cost (USD): ${cost_info['input_cost_usd']:.6f}")
        print(f"Output cost (USD): ${cost_info['output_cost_usd']:.6f}")
        print(f"Total cost (USD): ${cost_info['total_cost_usd']:.6f}")

        # Save cost info
        cost_df = pd.DataFrame([cost_info])
        cost_df.to_csv(f"data/batch_cost_info_{dataset_name}_resume.csv", index=False)

    # Print classification report if possible
    if label_col and label_col in df.columns:
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT (RESUMED)")
        print("=" * 60)
        print(classification_report(df[label_col], df["LLM alone"]))
    else:
        print("\nLabel column not available; skipping classification report.")

    print(f"\n✓ Resume processing complete for batch {batch_id} / dataset {dataset_name}")


if __name__ == "__main__":
    main()
