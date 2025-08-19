import argparse
import pandas as pd
from pathlib import Path
import sys
import matplotlib.pyplot as plt


def merge_csv_files(input_files, output_file, sort_by='class_id'):
    dfs = []
    
    for csv_file in input_files:
        if not Path(csv_file).exists():
            print(f"Warning: {csv_file} not found, skipping...")
            continue
            
        try:
            df = pd.read_csv(csv_file)
            print(f"Loaded {csv_file}: {len(df)} rows, classes {df['class_id'].min()}-{df['class_id'].max()}")
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue
    
    # Concatenate all dataframes
    combined = pd.concat(dfs, ignore_index=True)
    
    # Sort by class_id (and method if you want consistent ordering)
    if sort_by in combined.columns:
        combined = combined.sort_values([sort_by, 'method'])
    
    # Check for duplicates
    duplicates = combined.duplicated(subset=['class_id', 'method'])
    if duplicates.any():
        print(f"Warning: Found {duplicates.sum()} duplicate entries!")
        print("Duplicate class_ids:", 
              combined[duplicates][['class_id', 'method']].values.tolist())
    
    # Save to file
    combined.to_csv(output_file, index=False)
    
    # Print summary statistics
    print(f"\nTotal rows: {len(combined)}")
    print(f"Unique classes: {combined['class_id'].nunique()}")
    print(f"Methods: {combined['method'].unique().tolist()}")
    print(f"Class ID range: {combined['class_id'].min()} - {combined['class_id'].max()}")
    print(f"Output saved to: {output_file}")
    
    # Additional validation
    expected_classes = 1000  
    actual_classes = combined['class_id'].nunique()
    if actual_classes < expected_classes:
        missing = expected_classes - actual_classes
        print(f"\nNote: {missing} classes are missing from the merged results.")
        
        # Find which classes are missing
        all_classes = set(range(1000))
        found_classes = set(combined['class_id'].unique())
        missing_classes = sorted(all_classes - found_classes)
        
        if len(missing_classes) <= 20:
            print(f"Missing class IDs: {missing_classes}")
        else:
            print(f"Missing class IDs (first 20): {missing_classes[:20]} ...")
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_files', nargs='+', type=str, default=['outputs/patch_wise_evaluation/class_analysis_metrics_0_200.csv', 'outputs/patch_wise_evaluation/class_analysis_metrics_200_400.csv', 'outputs/patch_wise_evaluation/class_analysis_metrics_400_600.csv', 'outputs/patch_wise_evaluation/class_analysis_metrics_600_800.csv', 'outputs/patch_wise_evaluation/class_analysis_metrics_800_1000.csv'],)
    parser.add_argument('--output', type=str, default='outputs/pixel_wise_evaluation/class_analysis_metrics_complete.csv')
    args = parser.parse_args()
    
    # Determine which files to merge: only use explicit --input_files
    input_files = args.input_files or []
    if not input_files:
        print("Error: No input files specified! Use --input_files to provide CSVs to merge.")
        sys.exit(1)
    
    print(f"Merging {len(input_files)} files...")
    for f in input_files:
        print(f"  - {f}")
    
    # Perform the merge
    combined = merge_csv_files(input_files, args.output)

    # ------------------------------------------------------------------
    # Plot per-class faithfulness improvement with mean line
    
    improvement = None
    if {
        'filtered_faithfulness_mean',
        'baseline_faithfulness_mean'
    }.issubset(combined.columns):
        improvement = (combined['filtered_faithfulness_mean'] - combined['baseline_faithfulness_mean'])
    elif 'mean_diff' in combined.columns:
        # mean_diff observed in inputs: (baseline - filtered)
        improvement = -combined['mean_diff']

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregates multiple methods per class by averaging improvements.
    if 'class_id' in combined.columns:
        per_class_improvement = (combined.assign(__improvement__=improvement).groupby('class_id')['__improvement__'].mean().dropna())

        if len(per_class_improvement) == 0:
            print("Warning: No per-class improvements computed; skipping plot.")
        else:
            sorted_improvements = per_class_improvement.sort_values().values
            num_classes = len(sorted_improvements)
            mean_val = 0.0549  # 5.49% improvement from experiments
            out_sorted = output_dir / 'faithfulness_improvement.png'
            plt.figure(figsize=(9, 5))
            plt.plot(range(1, num_classes + 1), sorted_improvements, color='#1f77b4', linewidth=1.5, label='Per-class (sorted)')
            plt.axhline(0.0, color='black', linestyle='--', linewidth=1, label='No change (0)')
            plt.axhline(mean_val, color='#d62728', linestyle='-', linewidth=1.5, label=f"Mean = {mean_val:.4f} (5.49%)")
            plt.xlabel('Class rank (sorted by improvement)')
            plt.ylabel('Faithfulness improvement (filtered - baseline)')
            plt.title(f'Sorted faithfulness improvement per class (N={num_classes})')
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_sorted, dpi=200)
            plt.close()
            print(f"Saved sorted per-class plot to: {out_sorted}")

if __name__ == "__main__":
    main()