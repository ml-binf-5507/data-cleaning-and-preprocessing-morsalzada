
import argparse
import json
import os
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================================
# STEP 1: HANDLE MISSING VALUES & DUPLICATES
# ============================================================================

def replace_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace common missing value representations with NaN.
    
    Example:
        >>> df = pd.DataFrame({'A': ['NA', '1.5', 'N/A']})
        >>> df = replace_missing_values(df)
        >>> df['A'].isna().sum()
        2
    
    This function is PROVIDED as an example.
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Replace common missing value strings with np.nan
    missing_values = ["NA", "N/A", "na", "n/a", "NaN", "nan", ""]
    df = df.replace(missing_values, np.nan)
    
    return df


def remove_duplicates(df: pd.DataFrame, id_cols: List[str]) -> Tuple[pd.DataFrame, int]:
    """
    Remove duplicate rows, keeping id_cols unchanged.
    
    Returns:
        (cleaned_df, num_removed)
    
    Example:
        >>> df = pd.DataFrame({'id': [1,1,2], 'value': [10,10,20]})
        >>> df_clean, n = remove_duplicates(df, id_cols=['id'])
        >>> len(df_clean)
        2
    
    This function is PROVIDED as an example.
    """
    # Count how many duplicates we have
    num_duplicates = df.duplicated().sum()
    
    # Remove duplicate rows (keeps first occurrence)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    
    return df_clean, num_duplicates


# ============================================================================
# STEP 2: IDENTIFY FEATURE TYPES
# ============================================================================

def detect_feature_types(df: pd.DataFrame, target: str, id_cols: List[str]) -> Tuple[List[str], List[str]]:
    """
    Identify which columns are categorical vs numeric features.
    
    Example:
        >>> df = pd.DataFrame({'age': [25, 30], 'city': ['NYC', 'LA'], 'target': [0, 1]})
        >>> cat, num = detect_feature_types(df, target='target', id_cols=[])
        >>> cat
        ['city']
        >>> num
        ['age']
    """
    # TODO: Implement feature type detection
    # 1. Get all columns except target and id_cols:
    #    feature_cols = [c for c in df.columns if c not in id_cols and c != target]
    # 2. Identify categorical columns (dtype == 'object'):
    #    cat_cols = [c for c in feature_cols if df[c].dtype == 'object']
    # 3. Identify numeric columns (dtype in [int, float]):
    #    num_cols = [c for c in feature_cols if df[c].dtype in ['int64', 'float64']]
    # 4. Return (cat_cols, num_cols)
    pass


# ============================================================================
# STEP 3: ENCODE CATEGORICAL COLUMNS
# ============================================================================

def encode_categorical(df: pd.DataFrame, cat_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    One-hot encode categorical columns.
    
    Returns:
        (df_encoded, encoded_column_names)
    
    Example:
        >>> df = pd.DataFrame({'color': ['red', 'blue', 'red']})
        >>> df_enc, cols = encode_categorical(df, ['color'])
        >>> df_enc.columns.tolist()
        ['color_blue', 'color_red']
    
    IMPORTANT: This function is called separately on train and test data in run_preprocessing().
    You must ensure they produce the SAME columns. If test has a category not in train, 
    either exclude it or handle missing columns when encoding test.
    """
    # TODO: Implement one-hot encoding
    # 1. Create a copy of the dataframe
    # 2. For each column in cat_cols:
    #    a. Use pd.get_dummies(df[col], prefix=col, dtype=int) to one-hot encode
    #    b. Drop the original column: df.drop(col, axis=1, inplace=True)
    #    c. Add the new encoded columns: df = pd.concat([df, encoded], axis=1)
    # 3. Keep track of all new column names created
    # 4. Return (df_with_encoded_cols, list_of_encoded_column_names)
    #
    # HINT: When called in run_preprocessing(), you encode TRAIN first to get column names,
    # then when encoding TEST, you should only create those same columns (don't add new ones).
    # You can use pd.get_dummies(..., columns=...) or post-process to match columns.
    pass


# ============================================================================
# STEP 4: SCALE NUMERIC COLUMNS
# ============================================================================

def scale_numeric(df: pd.DataFrame, num_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    """
    Standardize numeric columns (mean=0, std=1).
    
    Returns:
        (df_scaled, means_dict, stds_dict)
    
    Example:
        >>> df = pd.DataFrame({'age': [20, 30, 40]})
        >>> df_scaled, means, stds = scale_numeric(df, ['age'])
        >>> abs(df_scaled['age'].mean()) < 0.001
        True
    """
    # TODO: Implement numeric scaling
    # 1. Create a copy of the dataframe
    # 2. For each column in num_cols:
    #    a. Handle missing values first: col.fillna(col.median())
    #    b. Calculate mean and std: mean = col.mean(), std = col.std()
    #    c. Standardize: (col - mean) / std
    # 3. Return (scaled_df, means_dict, stds_dict)
    pass


# ============================================================================
# MAIN PIPELINE (FULLY IMPLEMENTED - YOU DON'T NEED TO EDIT THIS!)
# ============================================================================

def run_preprocessing(
    input_path: str,
    target: str,
    output_dir: str = "outputs",
    id_cols: List[str] = None,
    impute_strategy: str = "median",
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Main preprocessing pipeline - split early to avoid data leakage.
    
    🎓 LEARNING NOTE: Pay attention to the comments marked with 🎓 throughout this
    function. They explain WHY we do things in this specific order. Understanding
    the workflow is just as important as implementing the transformations!
    
    Steps:
    1. Load data
    2. Replace "NA" string representations with NaN
    3. Remove duplicate rows
    4. Identify categorical vs numeric columns
    5. ⭐ **Split into train/test FIRST** ⭐ (crucial for avoiding leakage!)
    6. Impute missing values (using train parameters only)
    7. Encode categorical features (fit encoder on train, apply to both)
    8. Scale numeric features (fit scaler on train, apply to both)
    9. Save outputs and summary
    
    Args:
        input_path: Path to CSV file
        target: Name of target column
        output_dir: Where to save results
        id_cols: Column names to pass through unchanged (e.g., patient IDs)
        impute_strategy: "median", "mean", or "most_frequent"
        test_size: Fraction of data for testing (0.0 to 1.0)
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with summary of preprocessing steps
    
    🎓 WHY SPLIT FIRST? To prevent data leakage!
    Imagine you're building a model to predict house prices. If you compute the average
    price from ALL houses (including test), then use that average to fill missing values,
    you've "peeked" at the test set. Your model will look better than it really is!
    
    In real ML: test set = future data you haven't seen yet. Treat it that way!
    """
    if id_cols is None:
        id_cols = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load data
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    assert target in df.columns, f"Target column '{target}' not found in data"
    print(f"  Shape: {df.shape}")
    
    # Step 2: Replace missing value strings with NaN
    print("Handling missing values...")
    df = replace_missing_values(df)
    
    # Step 3: Remove duplicates
    print("Removing duplicates...")
    df, num_duplicates = remove_duplicates(df, id_cols)
    print(f"  Removed {num_duplicates} duplicate rows")
    
    # 🎓 CRITICAL DECISION POINT: When to split?
    # We split BEFORE computing any statistics (means, standard deviations, etc.)
    # 
    # WHY? Think about it: In real ML, the test set represents future data you haven't 
    # seen yet. If you compute statistics from ALL the data before splitting, you're
    # "peeking" at the test set. This is called DATA LEAKAGE and makes your model
    # look better than it really is!
    #
    # CORRECT ORDER: Split first → compute stats from train → apply to both train & test
    # WRONG ORDER: Compute stats from all data → split → apply (test data influenced the stats!)
    
    # Step 4: Identify feature types
    print("Identifying feature types...")
    
    # 🎓 LEARNING POINT: Why detect types from entire dataset (before split)?
    # Feature type detection (categorical vs numeric) is based on data types, not statistical
    # properties. It's OK to look at all data to determine types. However, ALL subsequent
    # statistical computations (means, categories, etc.) must use ONLY training data!
    
    cat_cols, num_cols = detect_feature_types(df, target, id_cols)
    print(f"  Categorical: {cat_cols}")
    print(f"  Numeric: {num_cols}")
    
    # Step 5: Split into train/test FIRST (before any statistics are computed!)
    print("Splitting into train/test (no data leakage!)...")
    feature_cols = cat_cols + num_cols
    X = df[feature_cols + id_cols].copy()
    y = df[target].copy()
    
    # Stratify to keep class balance
    stratify = y if len(y.unique()) <= 20 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 🎓 KEY PATTERN: Compute from train, apply to both
    # This pattern repeats throughout preprocessing:
    # 1. Compute a statistic from TRAIN (e.g., mean, encoding map, std dev)
    # 2. Apply it to TRAIN
    # 3. Apply the SAME statistic to TEST
    # This ensures test data doesn't influence our preprocessing decisions!
    
    # Step 6: Impute missing values (TRAIN parameters only!)
    print("  Imputing missing values (using TRAIN parameters)...")
    if num_cols:
        # Compute imputation values from TRAIN
        impute_values = {}
        for col in num_cols:
            if impute_strategy == "median":
                impute_values[col] = X_train[col].median()
            elif impute_strategy == "mean":
                impute_values[col] = X_train[col].mean()
        
        # 🎓 NOTICE: We computed impute_values from X_train only (above)
        # Now we apply those SAME values to both train and test
        # The test set doesn't get to influence what values we use!
        
        # Apply to both train and test
        for col in num_cols:
            X_train[col] = X_train[col].fillna(impute_values[col])
            X_test[col] = X_test[col].fillna(impute_values[col])
    
    # Also impute categorical with mode (from TRAIN)
    if cat_cols:
        for col in cat_cols:
            mode_val = X_train[col].mode()[0] if len(X_train[col].mode()) > 0 else X_train[col].iloc[0]
            X_train[col] = X_train[col].fillna(mode_val)
            X_test[col] = X_test[col].fillna(mode_val)
    
    # 🎓 DECISION POINT: Why encode AFTER splitting and imputing?
    # - After splitting: So we don't use test data to decide what categories exist
    # - After imputing: Because encode_categorical() expects no missing values
    # Order matters! Each step prepares data for the next step.
    
    # Step 7: Encode categorical features (TRAIN determines categories!)
    print("  Encoding categorical features (using TRAIN categories)...")
    encoded_columns = []
    if cat_cols:
        # Encode TRAIN first to get the column names
        X_train_encoded, train_encoded_cols = encode_categorical(X_train.copy(), cat_cols)
        encoded_columns = train_encoded_cols
        
        # Encode TEST with same columns
        X_test_encoded, _ = encode_categorical(X_test.copy(), cat_cols)
        
        # Ensure TEST has same columns as TRAIN (add missing, remove extra)
        for col in train_encoded_cols:
            if col not in X_test_encoded.columns:
                X_test_encoded[col] = 0  # Add missing column
        
        # Remove columns in TEST that aren't in TRAIN
        extra_cols = set(X_test_encoded.columns) - set(X_train_encoded.columns)
        X_test_encoded = X_test_encoded.drop(columns=list(extra_cols))
        
        # Reorder TEST columns to match TRAIN
        X_test_encoded = X_test_encoded[X_train_encoded.columns]
        
        X_train = X_train_encoded
        X_test = X_test_encoded
    
    # 🎓 FINAL TRANSFORMATION: Why scale LAST?
    # Scaling (standardization) is typically the final preprocessing step because:
    # 1. It works on numeric data only (so encoding must be done first)
    # 2. It's sensitive to outliers and missing values (so cleaning must be done first)
    # 3. Like all transformations, we compute mean/std from TRAIN and apply to both sets
    
    # Step 8: Scale numeric features (TRAIN parameters only!)
    print("  Scaling numeric features (using TRAIN parameters)...")
    scaled_columns = []
    train_means = {}
    train_stds = {}
    
    if num_cols:
        # Get numeric columns that still exist (not categorical)
        existing_num_cols = [c for c in num_cols if c in X_train.columns]
        
        if existing_num_cols:
            # Scale TRAIN and get parameters
            X_train_scaled, train_means, train_stds = scale_numeric(X_train.copy(), existing_num_cols)
            scaled_columns = existing_num_cols
            
            # Apply TRAIN parameters to TEST
            X_test_scaled = X_test.copy()
            for col in existing_num_cols:
                X_test_scaled[col] = (X_test[col] - train_means[col]) / train_stds[col]
            
            X_train = X_train_scaled
            X_test = X_test_scaled
    
    # 🎓 WORKFLOW COMPLETE! Let's review what happened:
    # 1. Load data → 2. Clean (duplicates, missing values) → 3. Detect types → 4. SPLIT
    # 5. Impute (compute from train, apply to both) → 6. Encode (compute from train, apply to both)
    # 7. Scale (compute from train, apply to both)
    # 
    # Key principle: Test set never influences preprocessing decisions!
    # This is how you prevent data leakage and build honest ML models.
    
    # Step 9: Save outputs
    print("Saving outputs...")
    
    # Combine features with id_cols and target
    train_output = X_train.copy()
    train_output['target'] = y_train.values
    
    test_output = X_test.copy()
    test_output['target'] = y_test.values
    
    # Save CSVs
    train_output.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_output.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    # 🎓 WHAT ARE WE SAVING?
    # - train.csv & test.csv: Cleaned, encoded, scaled data ready for ML
    # - summary.json: Statistics about the preprocessing (for validation/debugging)
    # In a real project, you'd also save the preprocessing parameters (means, std devs,
    # encoding maps) so you can apply the exact same transformations to new data later!
    
    # Create summary
    summary = {
        "input_file": input_path,
        "target": target,
        "duplicates_removed": num_duplicates,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "categorical_columns": cat_cols,
        "numeric_columns": num_cols,
        "encoded_columns": encoded_columns,
        "scaled_columns": scaled_columns,
        "feature_means_train": train_means,
        "feature_stds_train": train_stds
    }
    
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print("✓ Preprocessing complete!")
    print(f"  Outputs saved to: {output_dir}/")
    
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess a tabular dataset (beginner-friendly).")
    p.add_argument("--input", required=True, help="Path to input CSV file.")
    p.add_argument("--target", required=True, help="Target column name.")
    p.add_argument("--output-dir", default="outputs", help="Output directory (default: outputs).")
    p.add_argument("--id-cols", nargs="*", default=[], help="ID columns to keep unchanged (optional).")
    p.add_argument("--impute", default="median", choices=["median", "mean"], help="Imputation strategy (default: median).")
    p.add_argument("--test-size", type=float, default=0.2, help="Test fraction (default: 0.2).")
    p.add_argument("--random-state", type=int, default=42, help="Random seed (default: 42).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_preprocessing(
        input_path=args.input,
        target=args.target,
        output_dir=args.output_dir,
        id_cols=args.id_cols,
        impute_strategy=args.impute,
        random_state=args.random_state
    )
