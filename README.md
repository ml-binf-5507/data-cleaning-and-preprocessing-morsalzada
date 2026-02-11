[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/hv0wtSl-)
# Data Preprocessing Assignment

## Quick Start (Do This First)

1. **Look at working examples:**
   ```bash
   jupyter notebook TUTORIAL.ipynb
   ```

2. **Implement 3 functions** in `src/preprocess.py`:
   - `detect_feature_types()` - Identify categorical vs numeric columns
   - `encode_categorical()` - Convert categories to numbers
   - `scale_numeric()` - Standardize numeric features

3. **Test your code:**
   ```bash
   pytest tests/test_preprocessing.py -v
   ```

4. **Submit with AI disclosure:**
   - Complete [AI_DISCLOSURE.md](AI_DISCLOSURE.md)
   - Push to GitHub (tests run automatically)

---

## What You're Building

A data preprocessing pipeline with 3 steps:

1. **Clean data** - Handle missing values, remove duplicates (provided)
2. **Transform features** - Encode categories, scale numbers (you implement)
3. **Split for testing** - Train/test split to avoid data leakage (provided)

## The 3 Functions You Implement

### 1. `detect_feature_types(df, target, id_cols)` 
**Identify which columns are categorical vs numeric**

```python
# Categorical: dtype == 'object' (like "Toronto", "Vancouver")
# Numeric: dtype is int or float (like ages, prices)
# Return: (cat_cols_list, num_cols_list)
```

### 2. `encode_categorical(df, cat_cols)`
**Convert categories to numbers using one-hot encoding**

```python
# Input:  city=['Toronto', 'Vancouver']
# Output: city_Toronto=[1,0], city_Vancouver=[0,1]
# Use: pd.get_dummies(df[col], prefix=col, dtype=int)
```

### 3. `scale_numeric(df, num_cols)`
**Standardize numbers (mean=0, std=1)**

```python
# Formula: (value - mean) / std
# Returns: (scaled_df, means_dict, stds_dict)
```

---

## Testing

**Run all tests:**
```bash
pytest tests/test_preprocessing.py -v
```

**Test individual functions (for debugging):**
```bash
pytest tests/test_preprocessing.py::TestStudentFunctions -v
```

**Tests are organized by difficulty:**
- Phase 1: Basic (15 pts) - Does the pipeline run?
- Phase 2: Core (30 pts) - Is data cleaned correctly?
- Phase 3: Advanced (35 pts) - Are transformations correct?
- Phase 4: Pro (20 pts) - Is there data leakage?

**Minimum to pass:** Phase 1 + Phase 2 (45 pts)  
**Recommended:** Phase 1-3 (80 pts)  
**Full credit:** All phases (100 pts)

---

## Key Concept: Data Leakage ⚠️

**Never compute statistics from ALL data, then split.**

❌ **Wrong:**
```python
mean = df['age'].mean()  # Uses test data!
X_train, X_test = split(df)
```

✅ **Right:**
```python
X_train, X_test = split(df)
mean = X_train['age'].mean()  # Only train data
X_test['age'] = X_test['age'] - mean
```

**Why?** The test set represents "future data" you haven't seen. If you peek at it during preprocessing, your results will be wrong.

---

## Local Testing

**Test with sample data first:**
```bash
python src/preprocess.py \
  --input sample_data.csv \
  --target target
```

This creates `outputs/train.csv`, `outputs/test.csv`, `outputs/summary.json`

**Full example:**
```bash
python src/preprocess.py \
  --input your_data.csv \
  --target target_column \
  --output-dir outputs \
  --test-size 0.2 \
  --impute-strategy median
```

---

## Submission

Include in your repo:
- ✅ 3 functions implemented in `src/preprocess.py`
- ✅ `AI_DISCLOSURE.md` completed
- ✅ Passing tests (at least Phase 1 + Phase 2)

The tests run automatically when you push to GitHub.

