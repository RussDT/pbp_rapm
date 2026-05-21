import pandas as pd
import os
import glob
import pyarrow.parquet as pq

def analyze_files(directory):
    files = glob.glob(os.path.join(directory, "*.parquet"))
    results = []

    for file_path in sorted(files):
        try:
            file_name = os.path.basename(file_path)
            
            # Read schema to find all relevant columns efficiently
            parquet_file = pq.ParquetFile(file_path)
            cols = parquet_file.schema.names
            
            # Identify all target columns we want to average
            targets = []
            if 'Net_Diff' in cols: targets.append('Net_Diff')
            if 'Off_Diff' in cols: targets.append('Off_Diff')
            if 'Def_Diff' in cols: targets.append('Def_Diff')
            if 'Is_Rim_Attempt' in cols: targets.append('Is_Rim_Attempt')
            if 'Is_Turnover' in cols: targets.append('Is_Turnover')
            if 'Offensive_Rebound' in cols: targets.append('Offensive_Rebound')
            if 'Is_Rim_Make' in cols: targets.append('Is_Rim_Make')
            
            # Fallback if none of the above are found
            if not targets and len(cols) > 18:
                targets.append(cols[18])
            
            if targets:
                # Read only those columns from parquet
                df = pd.read_parquet(file_path, columns=targets)
                count = len(df)
                
                for target_col in targets:
                    avg = round(df[target_col].mean(), 3)
                    results.append({
                        "File": file_name,
                        "Event_Column": target_col,
                        "Average": avg,
                        "Count": count
                    })
                    print(f"Processed {file_name}: {target_col} Avg = {avg:.3f} (N={count})")
            else:
                print(f"Skipping {file_name}: No target column found.")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    return results

if __name__ == "__main__":
    processed_dir = "/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/processed/"
    print(f"Analyzing parquet files in {processed_dir}...")
    results = analyze_files(processed_dir)
    
    if results:
        summary_df = pd.DataFrame(results)
        output_path = "/Users/russellthomas/Docs/pbp_rapm/nba_pipeline/event_averages_summary.csv"
        summary_df.to_csv(output_path, index=False)
        print(f"\nSummary updated and saved to {output_path}")
    else:
        print("No results to save.")
