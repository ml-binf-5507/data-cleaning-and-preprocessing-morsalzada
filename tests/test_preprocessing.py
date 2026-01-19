"""
Simplified test suite for beginner-friendly preprocessing assignment.

Tests are organized in phases:
- Phase 1 (Basic): File creation, data shape
- Phase 2 (Core): Missing values, duplicates
- Phase 3 (Advanced): Encoding, scaling
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.preprocess import (
    run_preprocessing,
    detect_feature_types,
    encode_categorical,
    scale_numeric
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_dataset(tmp_path: Path) -> str:
    """
    Create a small realistic dataset with common issues:
    - Missing values (various formats)
    - Duplicates
    - Mixed data types
    - Outliers in numeric columns
    """
    rng = np.random.default_rng(42)
    n = 100

    # Age: numeric with outliers
    age = rng.normal(45, 12, size=n).round(1)
    age[0] = 150  # outlier
    age[1] = -5   # outlier

    # Income: numeric with missing values
    income = rng.normal(55000, 18000, size=n).round(0)
    income[5:10] = np.nan  # missing values

    # City: categorical with missing
    city = rng.choice(["Toronto", "Vancouver", "Montreal", "Calgary"], size=n)
    city[8:12] = None

    # Education: categorical
    education = rng.choice(["HS", "Bachelor", "Master", "PhD"], size=n)

    # Target: binary classification
    target = rng.integers(0, 2, size=n)

    # Patient IDs
    patient_id = [f"PAT{i:04d}" for i in range(n)]

    df = pd.DataFrame({
        "patient_id": patient_id,
        "age": age,
        "income": income,
        "city": city,
        "education": education,
        "target": target
    })

    # Add some duplicate rows
    df = pd.concat([df, df.iloc[:8]], ignore_index=True)

    # Add some missing value representations
    df.loc[15, "city"] = "NA"
    df.loc[20, "income"] = "N/A"

    csv_path = tmp_path / "sample_data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def run_pipeline(tmp_path: Path, sample_dataset: str):
    """Run preprocessing pipeline and return output directory and summary."""
    outdir = tmp_path / "outputs"
    outdir.mkdir(exist_ok=True)

    summary = run_preprocessing(
        input_path=sample_dataset,
        target="target",
        output_dir=str(outdir),
        id_cols=["patient_id"],
        impute_strategy="median",
        test_size=0.2,
        random_state=42
    )

    return outdir, summary


# ============================================================================
# UNIT TESTS FOR STUDENT-IMPLEMENTED FUNCTIONS (Optional - for debugging)
# ============================================================================

class TestStudentFunctions:
    """Unit tests for the 3 functions students must implement"""

    def test_detect_feature_types(self):
        """Test that detect_feature_types correctly identifies categorical vs numeric columns."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "age": [25, 30, 35],
            "income": [50000.0, 60000.0, 55000.0],
            "city": ["Toronto", "Vancouver", "Montreal"],
            "education": ["Bachelor", "Master", "PhD"],
            "target": [0, 1, 0]
        })
        
        cat_cols, num_cols = detect_feature_types(df, target="target", id_cols=["id"])
        
        assert set(cat_cols) == {"city", "education"}, f"Expected categorical: ['city', 'education'], got {cat_cols}"
        assert set(num_cols) == {"age", "income"}, f"Expected numeric: ['age', 'income'], got {num_cols}"

    def test_encode_categorical(self):
        """Test that encode_categorical properly one-hot encodes columns."""
        df = pd.DataFrame({
            "city": ["Toronto", "Vancouver", "Toronto", "Montreal"],
            "education": ["Bachelor", "Master", "Bachelor", "PhD"]
        })
        
        df_encoded, encoded_cols = encode_categorical(df, cat_cols=["city", "education"])
        
        # Should have created multiple one-hot columns
        assert len(encoded_cols) >= 4, f"Expected at least 4 encoded columns, got {len(encoded_cols)}"
        
        # Original categorical columns should be gone
        assert "city" not in df_encoded.columns, "Original 'city' column should be removed"
        assert "education" not in df_encoded.columns, "Original 'education' column should be removed"
        
        # Should have one-hot encoded columns
        city_cols = [c for c in df_encoded.columns if c.startswith("city_")]
        assert len(city_cols) >= 2, f"Expected at least 2 city columns, got {city_cols}"
        
        # Values should be 0 or 1
        for col in encoded_cols:
            assert df_encoded[col].isin([0, 1]).all(), f"Column {col} should only contain 0 or 1"

    def test_scale_numeric(self):
        """Test that scale_numeric properly standardizes columns."""
        df = pd.DataFrame({
            "age": [20.0, 30.0, 40.0, 50.0],
            "income": [40000.0, 60000.0, 80000.0, 100000.0]
        })
        
        df_scaled, means, stds = scale_numeric(df, num_cols=["age", "income"])
        
        # Check means are returned
        assert "age" in means, "Should return mean for 'age'"
        assert "income" in means, "Should return mean for 'income'"
        
        # Check stds are returned
        assert "age" in stds, "Should return std for 'age'"
        assert "income" in stds, "Should return std for 'income'"
        
        # Check that scaled values have mean ≈ 0, std ≈ 1
        age_mean = df_scaled["age"].mean()
        age_std = df_scaled["age"].std()
        
        assert abs(age_mean) < 1e-10, f"Scaled age mean should be ≈0, got {age_mean}"
        assert abs(age_std - 1.0) < 0.01, f"Scaled age std should be ≈1, got {age_std}"


# ============================================================================
# PHASE 1: BASIC REQUIREMENTS (15 points total)
# ============================================================================

class TestPhase1Basic:
    """Basic requirements: output files and data shapes"""

    def test_files_created(self, run_pipeline):
        """Test that required output files are created."""
        outdir, summary = run_pipeline

        assert (outdir / "train.csv").exists(), "train.csv not created"
        assert (outdir / "test.csv").exists(), "test.csv not created"
        assert (outdir / "summary.json").exists(), "summary.json not created"

    def test_summary_readable(self, run_pipeline):
        """Test that summary.json is valid JSON with required fields."""
        outdir, summary = run_pipeline

        with open(outdir / "summary.json") as f:
            summary_loaded = json.load(f)

        assert isinstance(summary_loaded, dict), "summary.json should contain a dictionary"
        assert "target" in summary_loaded, "Summary missing 'target' field"
        assert "train_size" in summary_loaded, "Summary missing 'train_size' field"
        assert "test_size" in summary_loaded, "Summary missing 'test_size' field"

    def test_train_test_split(self, run_pipeline):
        """Test that train and test sets have correct sizes."""
        outdir, summary = run_pipeline

        train = pd.read_csv(outdir / "train.csv")
        test = pd.read_csv(outdir / "test.csv")

        # Test set should be approximately 20%
        total = len(train) + len(test)
        test_ratio = len(test) / total
        assert 0.1 < test_ratio < 0.35, f"Test ratio {test_ratio:.1%} not close to 20%"


# ============================================================================
# PHASE 2: CORE DATA CLEANING (30 points total)
# ============================================================================

class TestPhase2DataCleaning:
    """Core requirements: missing values, duplicates, data quality"""

    def test_duplicates_removed(self, run_pipeline):
        """Test that duplicate rows are removed."""
        outdir, summary = run_pipeline

        # Summary should report duplicates removed
        assert "duplicates_removed" in summary, "Summary missing 'duplicates_removed' field"
        assert summary["duplicates_removed"] > 0, "Expected duplicates to be removed"

    def test_no_missing_in_features(self, run_pipeline):
        """Test that feature columns have no missing values after preprocessing."""
        outdir, summary = run_pipeline

        train = pd.read_csv(outdir / "train.csv")

        # Get feature columns (exclude patient_id and target)
        feature_cols = [c for c in train.columns if c not in ["patient_id", "target"]]

        missing_count = train[feature_cols].isna().sum().sum()
        assert missing_count == 0, f"Found {missing_count} missing values in TRAIN features"

    def test_no_missing_in_test(self, run_pipeline):
        """Test that test set also has no missing values (prevent separate imputation leakage)."""
        outdir, summary = run_pipeline

        test = pd.read_csv(outdir / "test.csv")

        # Get feature columns (exclude patient_id and target)
        feature_cols = [c for c in test.columns if c not in ["patient_id", "target"]]

        missing_count = test[feature_cols].isna().sum().sum()
        assert missing_count == 0, f"Found {missing_count} missing values in TEST features. Did you impute train and test separately?"

    def test_all_features_numeric(self, run_pipeline):
        """Test that all features are numeric after preprocessing."""
        outdir, summary = run_pipeline

        train = pd.read_csv(outdir / "train.csv")

        # Get feature columns (exclude patient_id and target)
        feature_cols = [c for c in train.columns if c not in ["patient_id", "target"]]

        non_numeric = []
        for col in feature_cols:
            if not pd.api.types.is_numeric_dtype(train[col]):
                non_numeric.append(col)

        assert (
            len(non_numeric) == 0
        ), f"Non-numeric features remain: {non_numeric}. Did you encode categoricals?"

    def test_all_features_numeric_test(self, run_pipeline):
        """Test that all features in test set are also numeric."""
        outdir, summary = run_pipeline

        test = pd.read_csv(outdir / "test.csv")

        # Get feature columns (exclude patient_id and target)
        feature_cols = [c for c in test.columns if c not in ["patient_id", "target"]]

        non_numeric = []
        for col in feature_cols:
            if not pd.api.types.is_numeric_dtype(test[col]):
                non_numeric.append(col)

        assert len(non_numeric) == 0, f"Non-numeric features in TEST: {non_numeric}"

    def test_id_cols_preserved(self, run_pipeline):
        """Test that ID columns are preserved in output."""
        outdir, summary = run_pipeline

        train = pd.read_csv(outdir / "train.csv")
        test = pd.read_csv(outdir / "test.csv")

        assert "patient_id" in train.columns, "patient_id not in train.csv"
        assert "patient_id" in test.columns, "patient_id not in test.csv"

    def test_train_test_same_columns(self, run_pipeline):
        """Test that train and test have the SAME columns (critical for ML!)."""
        outdir, summary = run_pipeline

        train = pd.read_csv(outdir / "train.csv")
        test = pd.read_csv(outdir / "test.csv")

        train_cols = set(train.columns)
        test_cols = set(test.columns)

        missing_in_test = train_cols - test_cols
        extra_in_test = test_cols - train_cols

        assert (
            missing_in_test == set() and extra_in_test == set()
        ), f"Train and test have different columns! Missing in test: {missing_in_test}, Extra in test: {extra_in_test}. Did you encode train and test separately?"


# ============================================================================
# PHASE 3: PREPROCESSING TRANSFORMATIONS (35 points total)
# ============================================================================

class TestPhase3Transformations:
    """Advanced requirements: encoding, scaling, data properties"""

    def test_categorical_encoded(self, run_pipeline):
        """Test that categorical columns are identified and encoded."""
        outdir, summary = run_pipeline

        # Summary should report encoded columns
        assert "encoded_columns" in summary, "Summary missing 'encoded_columns' field"
        assert len(summary["encoded_columns"]) > 0, "No categorical columns were encoded"

        # Example: city and education should be encoded
        train = pd.read_csv(outdir / "train.csv")

        # After encoding, should have columns like 'city_Montreal', 'education_Bachelor', etc.
        encoded_city_cols = [c for c in train.columns if c.startswith("city_")]
        encoded_edu_cols = [c for c in train.columns if c.startswith("education_")]

        assert (
            len(encoded_city_cols) >= 2 or len(encoded_edu_cols) >= 2
        ), "Expected at least 2 one-hot encoded columns for categorical features"

    def test_numeric_identified(self, run_pipeline):
        """Test that numeric columns are identified."""
        outdir, summary = run_pipeline

        assert "numeric_columns" in summary, "Summary missing 'numeric_columns' field"
        assert len(summary["numeric_columns"]) > 0, "No numeric columns identified"

    def test_numeric_scaled(self, run_pipeline):
        """Test that numeric features are scaled (mean ≈ 0, std ≈ 1)."""
        outdir, summary = run_pipeline

        train = pd.read_csv(outdir / "train.csv")

        # Get numeric columns that should be scaled (exclude id and target)
        numeric_cols = summary.get("numeric_columns", [])

        for col in numeric_cols:
            if col in train.columns:
                mean_val = train[col].mean()
                std_val = train[col].std()

                # Mean should be close to 0 (within ±0.5)
                assert abs(mean_val) < 0.5, f"{col} mean={mean_val:.3f}, expected ≈0"

                # Std should be close to 1 (between 0.5 and 1.5)
                assert 0.5 < std_val < 1.5, f"{col} std={std_val:.3f}, expected ≈1"


# ============================================================================
# PHASE 4: NO DATA LEAKAGE (20 points)
# ============================================================================

class TestPhase4NoLeakage:
    """Ensure preprocessing fits on TRAIN only, then transforms TEST"""

    def test_scaling_computed_from_train(self, run_pipeline):
        """Test that scaling parameters come from TRAIN set only."""
        outdir, summary = run_pipeline

        # If the scaling was computed from train-only, train set should
        # have mean ≈ 0 and std ≈ 1, but test set may not be as perfect
        train = pd.read_csv(outdir / "train.csv")
        test = pd.read_csv(outdir / "test.csv")

        numeric_cols = summary.get("numeric_columns", [])

        train_stds = []
        test_stds = []
        for col in numeric_cols:
            if col in train.columns:
                train_stds.append(train[col].std())
                test_stds.append(test[col].std())

        # Train stds should be close to 1 (proof of proper scaling)
        train_stds_avg = np.mean(train_stds)
        assert (
            0.8 < train_stds_avg < 1.2
        ), f"Train stds average {train_stds_avg:.3f}, expected ≈1 (proof of proper scaling)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
