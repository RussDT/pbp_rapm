import subprocess
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import time

def run_process_rapm(input_file):
    """Function to run the processing script for a single file."""
    script_path = "nba_pipeline/scripts/02_process_rapm.py"
    # We use absolute paths to be safe when running in parallel
    abs_script_path = Path(script_path).absolute()
    abs_input_file = Path(input_file).absolute()
    
    cmd = ["python", str(abs_script_path), str(abs_input_file)]
    
    print(f"Starting: {input_file}")
    start_time = time.time()
    
    # Run the command. We capture output to avoid messy interleaving in the terminal,
    # then print it all at once when finished.
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    end_time = time.time()
    duration = end_time - start_time
    
    if result.returncode == 0:
        status = f"SUCCESS: Processed {input_file} in {duration:.2f}s"
        # Optional: uncomment to see full output for each process
        # print(result.stdout)
    else:
        status = f"ERROR: Failed to process {input_file} in {duration:.2f}s"
        print(f"\n--- Error Output for {input_file} ---")
        print(result.stdout)
        print(result.stderr)
        print("-" * 40)
        
    print(status)
    return status

def main():
    # Base directory for raw data
    raw_data_dir = Path("nba_pipeline/raw_data")
    
    # List of seasons and playoff years to process
    seasons = ["23", "24", "25", "26"]
    suffixes = ["", "_PS"]
    
    files_to_process = []
    for season in seasons:
        for suffix in suffixes:
            filename = f"NBA{season}{suffix}.parquet"
            file_path = raw_data_dir / filename
            
            if file_path.exists():
                files_to_process.append(str(file_path))
            else:
                print(f"Warning: File {file_path} not found.")

    num_files = len(files_to_process)
    print(f"Found {num_files} files to process.")
    
    # Determine number of workers. 
    # NBA processing is memory intensive, so we limit parallel processes 
    # to avoid OOM (Out of Memory) errors on large files.
    # Defaulting to 4 workers, or fewer if there are fewer files.
    max_workers = min(4, num_files, os.cpu_count() or 1)
    
    print(f"Starting parallel processing with {max_workers} workers...\n")
    
    start_all = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(run_process_rapm, files_to_process))
    
    end_all = time.time()
    
    print(f"\n{'='*60}")
    print(f"All processing complete in {(end_all - start_all)/60:.2f} minutes.")
    print(f"{'='*60}")
    for res in results:
        print(res)

if __name__ == "__main__":
    main()
