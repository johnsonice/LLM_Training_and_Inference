from sklearn.model_selection import train_test_split
import pandas as pd
def train_val_test_split(dataset, val_test_ratios=0.3, label_col='label', RANDOM_SEED=42, verbose=False):
    """
    Split dataset into train, validation and test sets with stratification.
    
    Args:
        dataset: Input pandas DataFrame
        val_test_ratios: Combined ratio for validation and test sets (default 0.3)
        label_col: Name of the label column for stratification (default 'label')
        RANDOM_SEED: Random seed for reproducibility (default 42)
        verbose: Whether to print split sizes (default False)
    
    Returns:
        Dictionary containing train, validation and test splits
    """
    # Calculate split ratios
    test_size = val_test_ratios / 2
    val_size = val_test_ratios / 2
    
    # First split into train and temp (val+test)
    train_df, temp_df = train_test_split(
        dataset,
        test_size=val_test_ratios, 
        stratify=dataset[label_col],
        random_state=RANDOM_SEED
    )
    
    # Split temp into val and test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df[label_col], 
        random_state=RANDOM_SEED
    )
    
    # Create final split dictionary
    splits = {
        'train': train_df,
        'validation': val_df,
        'test': test_df
    }
    
    # Print split sizes if verbose
    if verbose:
        print(f"Train size: {len(splits['train'])}")
        print(f"Validation size: {len(splits['validation'])}")
        print(f"Test size: {len(splits['test'])}")
    
    return splits

def load_split_climate_data(data_path, merge_neutral=False, verbose=False):
    """
    Load climate news data, map sentiment labels, and split into train/val/test sets
    
    Args:
        data_path (str): Path to the CSV file containing climate news data
        merge_neutral (bool): If True, merges neutral with favorable labels
        verbose (bool): If True, prints label distribution (default False)
        
    Returns:
        dict: Dictionary containing train, validation and test splits
    """
    # Read and preprocess data
    df = pd.read_csv(data_path)
    keep_cols = ['title', 'body', 'Rating']
    df = df[keep_cols]
    df.columns = ['title', 'paragraph', 'label']
    
    # Map numeric labels to text
    if merge_neutral:
        sentiment_map = {
            -1: "unfavorable",
            0: "favorable",  # Neutral merged with favorable
            1: "favorable"
        }
    else:
        sentiment_map = {
            -1: "unfavorable",
            0: "neutral",
            1: "favorable"
        }
    
    df['label'] = df['label'].map(sentiment_map)
    
    # Print label distribution if verbose
    if verbose:
        print('Label distribution:')
        print(df['label'].value_counts(normalize=True))
    
    # Split data
    splits = train_val_test_split(
        df, 
        val_test_ratios=0.3,
        label_col='label',
        RANDOM_SEED=42
    )
    
    return splits

