# Quick Reference: Pandas One-Liners for This Assignment

## Missing Values
```python
# Replace strings with NaN
df = df.replace(["NA", "N/A", "na", "NaN", ""], np.nan)

# Impute with median
df['age'] = df['age'].fillna(df['age'].median())

# Impute with mode (most frequent)
df['city'] = df['city'].fillna(df['city'].mode()[0])

# Check missing count
print(df.isna().sum())
```

## Data Types & Filtering
```python
# Check dtypes
print(df.dtypes)

# Get numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Get categorical columns
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

# Filter columns (exclude ID and target)
features = [c for c in df.columns if c not in ['patient_id', 'target']]
```

## Duplicates
```python
# Count duplicates
n_dup = df.duplicated().sum()

# Remove duplicates
df = df.drop_duplicates().reset_index(drop=True)
```

## One-Hot Encoding
```python
# One-hot encode one column
encoded = pd.get_dummies(df['city'], prefix='city', dtype=int)

# Replace original with encoded
df = df.drop('city', axis=1)
df = pd.concat([df, encoded], axis=1)

# Check result
print(df.columns)  # Should see city_Calgary, city_Montreal, etc.
```

## Scaling (Standardization)
```python
# Compute mean and std
mean = df['age'].mean()
std = df['age'].std()

# Standardize: (x - mean) / std
df['age'] = (df['age'] - mean) / std

# Verify: mean should be ≈ 0, std ≈ 1
print(f"Mean: {df['age'].mean()}")
print(f"Std: {df['age'].std()}")
```

## Train/Test Split
```python
from sklearn.model_selection import train_test_split

# Prepare X, y
X = df.drop(['patient_id', 'target'], axis=1)
y = df['target']

# Split (stratify keeps class distribution balanced)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
```

## Scaling on Train Parameters Only (No Leakage!)
```python
# CORRECT approach
# 1. Compute from train
train_mean = X_train['age'].mean()
train_std = X_train['age'].std()

# 2. Apply to train
X_train['age'] = (X_train['age'] - train_mean) / train_std

# 3. Apply SAME parameters to test
X_test['age'] = (X_test['age'] - train_mean) / train_std
```

## File I/O
```python
# Read CSV
df = pd.read_csv('data.csv')

# Save CSV
df.to_csv('output.csv', index=False)

# Save JSON
import json
summary = {"key": "value"}
with open('summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
```

## Concatenation
```python
# Combine dataframes side-by-side (add columns)
df = pd.concat([df, encoded], axis=1)

# Combine dataframes top-to-bottom (add rows)
df = pd.concat([df1, df2], axis=0)
```

## Debugging
```python
# Print shape
print(f"Shape: {df.shape}")  # (rows, columns)

# Print first N rows
print(df.head(10))

# Print column names
print(df.columns.tolist())

# Print basic stats
print(df.describe())

# Check specific column
print(df['age'].unique())
print(df['age'].value_counts())
```
