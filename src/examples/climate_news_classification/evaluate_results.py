import pandas as pd
import argparse
from pathlib import Path
from collections import defaultdict

def analyze_file(file_path: Path) -> dict:
    """Analyze a single results file for errors and None values."""
    df = pd.read_csv(file_path)
    none_count = df['llm_predicted_label'].isna().sum()
    error_count = df['llm_justification'].str.startswith('Error:').sum() if 'llm_justification' in df.columns else 0
    
    return {
        'file_name': file_path.name,
        'total_rows': len(df),
        'none_count': none_count,
        'error_count': error_count,
        'label_distribution': df['llm_predicted_label'].value_counts().to_dict(),
        'error_rate': (none_count + error_count) / len(df) if len(df) > 0 else 0
    }

def generate_summary(analyses):
    """Generate summary statistics from analysis results."""
    summary_df = pd.DataFrame(analyses)
    total_rows = sum(a['total_rows'] for a in analyses)
    
    # Aggregate label distribution
    label_dist = defaultdict(int)
    for a in analyses:
        for label, count in a['label_distribution'].items():
            if pd.notna(label):
                label_dist[label] += count
    
    total_errors = sum(a['none_count'] + a['error_count'] for a in analyses)
    return summary_df, {
        'total_files': len(analyses),
        'total_rows': total_rows,
        'total_errors': total_errors,
        'overall_error_rate': total_errors / total_rows if total_rows else 0,
        'label_distribution': dict(label_dist)
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM classification results')
    parser.add_argument('--results-dir', type=str, 
                       default='/ephemeral/home/xiong/data/Fund/Climate/batch_tasks_results_v1_no_formated_output')
    parser.add_argument('--error-threshold', type=float, default=0.01)
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    # Analyze files
    analyses = []
    for file_path in results_dir.glob('results_*.csv'):
        try:
            analyses.append(analyze_file(file_path))
        except Exception as e:
            print(f"Error analyzing {file_path.name}: {str(e)}")
    
    if not analyses:
        print("No files were processed.")
        return
    
    # Generate summary and print stats
    summary_df, stats = generate_summary(analyses)
    print("\n=== Aggregate Statistics ===")
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Handle problematic files
    problematic = summary_df[summary_df['error_rate'] > args.error_threshold]['file_name'].tolist()
    if problematic:
        print(f"\nFound {len(problematic)} files with error rate > {args.error_threshold}:")
        print("\n".join(f"- {f}" for f in problematic))
        
        if input("\nDelete these files? (yes/no): ").lower() == 'yes':
            for file in problematic:
                try:
                    (results_dir / file).unlink()
                    print(f"Deleted {file}")
                except Exception as e:
                    print(f"Error deleting {file}: {str(e)}")
    else:
        print("\nNo problematic files found.")
    
    # Save summary
    summary_df.to_csv(results_dir / 'evaluation_summary.csv', index=False)
    print(f"\nSummary saved to: {results_dir / 'evaluation_summary.csv'}")

if __name__ == "__main__":
    main() 