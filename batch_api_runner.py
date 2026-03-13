import os
import json
import time
import pandas as pd
import dotenv
from openai import OpenAI
import jsonlines
from sklearn.metrics import classification_report
from datetime import datetime

def save_batch_metadata(batch_id, file_id, dataset_name, managed_run_info, timestamp):
    """Save batch metadata for later retrieval"""
    metadata = {
        "batch_id": batch_id,
        "file_id": file_id,
        "dataset_name": dataset_name,
        "timestamp": timestamp,
        "model": managed_run_info.get("model", "gpt-5-mini"),
        "managed_run_info": managed_run_info,
        "status": "submitted"
    }
    
    # Load existing metadata if it exists
    metadata_file = "batch_metadata.json"
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            all_metadata = json.load(f)
    else:
        all_metadata = {}
    
    # Store this batch's metadata
    all_metadata[batch_id] = metadata
    
    # Save back to file
    with open(metadata_file, "w") as f:
        json.dump(all_metadata, f, indent=2)
    
    print(f"Batch metadata saved to {metadata_file}")
    return metadata_file

def load_batch_metadata(batch_id=None):
    """Load batch metadata, optionally filtered by batch_id"""
    metadata_file = "batch_metadata.json"
    if not os.path.exists(metadata_file):
        print(f"No metadata file found at {metadata_file}")
        return None
    
    with open(metadata_file, "r") as f:
        all_metadata = json.load(f)
    
    if batch_id:
        return all_metadata.get(batch_id)
    else:
        return all_metadata

def retrieve_batch_results_by_id(batch_id):
    """Retrieve results for a specific batch ID without resubmitting"""
    metadata = load_batch_metadata(batch_id)
    if not metadata:
        print(f"No metadata found for batch {batch_id}")
        return None
    
    OPENAI_API_KEY = dotenv.get_key(dotenv.find_dotenv(), "API_KEY")
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    print(f"\nRetrieving results for batch {batch_id}...")
    print(f"Dataset: {metadata['dataset_name']}")
    print(f"Submitted: {metadata['timestamp']}")
    
    batch_results, batch = retrieve_batch_results(client, batch_id)
    
    return batch_results, batch, metadata

def create_batch_requests(df, ti, ab, my_prompt, model="gpt-5-mini", ti_col="title_text", ab_col="abstract_text"):
    """Create batch requests in JSONL format"""
    requests = []
    for i, row in df.iterrows():
        ti_text = row.get(ti, "")
        ab_text = row.get(ab, "")
        ti_abs_key = "{} {}".format(ti_text, ab_text).strip()
        prompt = "{} {}".format(my_prompt, ti_abs_key)
        
        request = {
            "custom_id": str(i),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
        }
        requests.append(request)
    
    return requests

def save_batch_file(requests, filename="batch_requests.jsonl"):
    """Save requests to JSONL file"""
    with jsonlines.open(filename, mode='w') as writer:
        for request in requests:
            writer.write(request)
    return filename

def submit_batch(client, batch_file):
    """Submit batch to OpenAI"""
    with open(batch_file, "rb") as f:
        batch_response = client.files.create(
            file=f,
            purpose="batch"
        )
    
    file_id = batch_response.id
    print(f"File uploaded with ID: {file_id}")
    
    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    
    print(f"Batch submitted with ID: {batch.id}")
    return batch.id, file_id

def poll_batch_completion(client, batch_id, check_interval=60, max_wait_time=10000):
    """Poll batch status until completion"""
    start_time = time.time()
    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"Batch {batch_id} status: {batch.status}")
        # Newer Batch objects expose request counts via batch.request_counts
        try:
            rc = batch.request_counts
            in_progress = getattr(rc, "in_progress", None)
            completed = getattr(rc, "completed", None)
            failed = getattr(rc, "failed", None)
            details = []
            if in_progress is not None:
                details.append(f"In progress: {in_progress}")
            if completed is not None:
                details.append(f"Completed: {completed}")
            if failed is not None:
                details.append(f"Failed: {failed}")
            if details:
                print("  " + ", ".join(details))
        except AttributeError:
            # Older/other SDK versions may not have request_counts; skip detailed counts
            pass
        
        if batch.status == "completed":
            print(f"Batch completed!")
            return batch
        
        if batch.status == "failed":
            print(f"Batch failed!")
            return batch
        
        elapsed = time.time() - start_time
        if elapsed > max_wait_time:
            print(f"Batch did not complete within {max_wait_time} seconds")
            return batch
        
        time.sleep(check_interval)

def retrieve_batch_results(client, batch_id):
    """Retrieve results from completed batch"""
    batch = client.batches.retrieve(batch_id)
    
    if batch.status != "completed":
        print(f"Warning: Batch status is {batch.status}, not completed; results file may not be ready yet.")
        # When batch is still running, output_file_id is None and attempting to
        # fetch content will fail. Let the caller decide what to do.
        return [], batch

    result_file_id = batch.output_file_id
    if not result_file_id:
        print("No output_file_id available even though batch is completed.")
        return [], batch

    results = client.files.content(result_file_id).text
    
    # Parse JSONL results
    batch_results = []
    for line in results.split('\n'):
        if line.strip():
            batch_results.append(json.loads(line))
    
    return batch_results, batch

def process_batch_results(df, batch_results):
    """Process batch results and create predictions"""
    # Create mapping of custom_id to result
    results_map = {}
    for result in batch_results:
        custom_id = result.get("custom_id")
        if result.get("error"):
            results_map[custom_id] = {
                "prediction": 0,
                "justification": f"Error: {result['error'].get('message', 'Unknown error')}",
                "raw_response": str(result)
            }
        else:
            openai_response = result["response"]["body"]["choices"][0]["message"]["content"]
            results_map[custom_id] = {
                "raw_response": openai_response,
                "justification": openai_response.replace("\n", " ").replace("  ", " ")
            }
    
    # Process predictions based on responses
    predictions = []
    justifications = []
    raw_responses = []
    
    for i in range(len(df)):
        result = results_map.get(str(i))
        if result:
            raw_responses.append(result["raw_response"])
            justifications.append(result["justification"])
            
            # Apply same logic as original script
            openai_response = result["raw_response"]
            if openai_response.lower().startswith("yes") or openai_response.lower().startswith("**yes**") or "yes" in openai_response.lower()[:10]:
                predictions.append(1)
            else:
                predictions.append(0)
        else:
            predictions.append(0)
            justifications.append("No response")
            raw_responses.append("No response")
    
    return predictions, justifications, raw_responses

def calculate_batch_cost(batch, model="gpt-5-mini"):
    """Calculate cost from batch metadata"""
    # Pricing for different models (as of March 2026)
    pricing = {
        "gpt-4o-mini": {
            "input": 0.15,      # per 1M tokens
            "output": 0.60      # per 1M tokens
        },
        "gpt-5-mini": {
            "input": 0.075,     # per 1M tokens - 50% cheaper than gpt-4o-mini
            "output": 0.30      # per 1M tokens
        }
    }
    
    if model not in pricing:
        print(f"Warning: Model {model} not in pricing table. Using gpt-5-mini pricing.")
        model = "gpt-5-mini"
    
    rates = pricing[model]
    input_tokens = batch.request_counts.completed * 100  # Approximate estimate
    output_tokens = batch.request_counts.completed * 50   # Approximate estimate
    
    input_cost = (input_tokens / 1_000_000) * rates["input"]
    output_cost = (output_tokens / 1_000_000) * rates["output"]
    total_cost = input_cost + output_cost
    
    return {
        "completed_requests": batch.request_counts.completed,
        "failed_requests": batch.request_counts.failed,
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": total_cost
    }

if __name__ == '__main__':
    dotenv.load_dotenv()
    
    # ==================== MODEL SELECTION ====================
    # Choose which model to use for batch processing
    # Options: "gpt-4o-mini" or "gpt-5-mini"
    selected_model = "gpt-5-mini"  # Change this to switch models
    # ========================================================
    
    outfolder = r"data\\LLM_predictions"
    managed_runs = json.load(open(r"managed_runs.json", "r"))
    
    OPENAI_API_KEY = dotenv.get_key(dotenv.find_dotenv(), "API_KEY")
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    #my_dataset = "MRC, BPA"  # Change this to process different datasets
    done_data = ["BPA"]
    
    print(f"\n{'='*60}")
    print(f"BATCH API PROCESSING")
    print(f"Selected Model: {selected_model}")
    print(f"{'='*60}\n")
    
    for my_dataset in managed_runs.keys():
        if my_dataset not in done_data:
            continue
        
        print(f"\n\n{'='*60}")
        print(f"Running batch API for dataset: {my_dataset}")
        print(f"{'='*60}\n")
        
        filename = managed_runs[my_dataset]["path"]
        ti = managed_runs[my_dataset]["title_col"]
        ab = managed_runs[my_dataset]["abstract_col"]
        my_prompt = managed_runs[my_dataset]["prompt"]
        label_col = managed_runs[my_dataset]["label_col"]
        
        # Read data
        try:
            df = pd.read_csv(filename, encoding='utf-8').fillna("")
            enc = 'utf-8'
        except:
            df = pd.read_csv(filename, encoding='windows-1252').fillna("")
            enc = 'windows-1252'
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        seed = 0
        df = df.sample(frac=1, random_state=seed)
        df.reset_index(drop=True, inplace=True)
        
        # Create batch requests
        print("\nCreating batch requests...")
        requests = create_batch_requests(df, ti, ab, my_prompt, model=selected_model)
        batch_file = save_batch_file(requests, f"batch_requests_{my_dataset}.jsonl")
        print(f"Saved {len(requests)} requests to {batch_file}")
        
        # Submit batch
        print("\nSubmitting batch...")
        batch_id, file_id = submit_batch(client, batch_file)
        
        # Save batch metadata for later retrieval
        timestamp = datetime.now().isoformat()
        managed_run_info = {
            "dataset": my_dataset,
            "title_col": ti,
            "abstract_col": ab,
            "label_col": label_col,
            "num_records": len(df),
            "model": selected_model
        }
        save_batch_metadata(batch_id, file_id, my_dataset, managed_run_info, timestamp)
        
        # Poll for completion
        print("\nPolling for batch completion (this may take a while)...")
        completed_batch = poll_batch_completion(client, batch_id)
        
        # Retrieve results
        print("\nRetrieving results...")
        batch_results, batch_info = retrieve_batch_results(client, batch_id)
        
        # Process results
        print("\nProcessing results...")
        predictions, justifications, raw_responses = process_batch_results(df, batch_results)
        
        # Add to dataframe
        df["LLM alone"] = predictions
        df["LLM Justification"] = justifications
        df["LLM Raw Response"] = raw_responses
        
        # Save outputs
        print("\nSaving outputs...")
        df.to_csv(os.path.join(outfolder, filename.split("\\")[-1]), index=False, encoding=enc)
        df.to_csv("data/batch_backup.csv", index=False, encoding=enc)
        
        # Save raw responses
        raw_df = pd.DataFrame({
            "index": range(len(raw_responses)),
            "raw_response": raw_responses
        })
        raw_df.to_csv(f"data/batch_raw_responses_{my_dataset}.csv", index=False, encoding=enc)
        
        # Calculate and display costs
        print("\n" + "="*60)
        print("BATCH COST ANALYSIS")
        print("="*60)
        print(f"Model: {selected_model}")
        cost_info = calculate_batch_cost(batch_info, model=selected_model)
        print(f"Completed requests: {cost_info['completed_requests']}")
        print(f"Failed requests: {cost_info['failed_requests']}")
        print(f"Estimated input tokens: {cost_info['estimated_input_tokens']}")
        print(f"Estimated output tokens: {cost_info['estimated_output_tokens']}")
        print(f"Input cost (USD): ${cost_info['input_cost_usd']:.6f}")
        print(f"Output cost (USD): ${cost_info['output_cost_usd']:.6f}")
        print(f"Total cost (USD): ${cost_info['total_cost_usd']:.6f}")
        print(f"\nBatch ID for retrieval: {batch_id}")
        print(f"Save this ID if you need to retrieve results later!")
        
        # Save cost info
        cost_df = pd.DataFrame([cost_info])
        cost_df.to_csv(f"data/batch_cost_info_{my_dataset}.csv", index=False)
        
        # Print classification report
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(df[label_col], df["LLM alone"]))
        
        # Clean up uploaded file
        print("\nCleaning up...")
        client.files.delete(file_id)
        os.remove(batch_file)
        
        print(f"✓ Batch processing complete for {my_dataset}")

# ==========================================================================================================
# USAGE FOR RETRIEVING RESULTS LATER
# ==========================================================================================================
# If a batch is still processing or you need to retrieve results at a later time, use the functions below:
#
# 1. List all stored batch metadata:
#    all_batches = load_batch_metadata()
#    print(all_batches)
#
# 2. Retrieve a specific batch:
#    batch_id = "batch_XXXXXXXXXXXXXXXXXX"  # The ID shown when the batch was submitted
#    batch_results, batch_info, metadata = retrieve_batch_results_by_id(batch_id)
#
# 3. Check batch status in OpenAI dashboard: https://platform.openai.com/batches
#
# The batch metadata is stored in batch_metadata.json for your records.
# ==========================================================================================================

