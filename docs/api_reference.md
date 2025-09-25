# API Reference

## Data Preprocessing Module

### `DataPreprocessor` Class

#### `load_data()`
Load raw data from CSV file.

**Returns:**
- `pd.DataFrame`: Raw dataset

#### `clean_data(df)`
Clean and preprocess the data.

**Parameters:**
- `df` (pd.DataFrame): Raw data

**Returns:**
- `pd.DataFrame`: Cleaned data

#### `prepare_data(df)`
Prepare data for modeling.

**Parameters:**
- `df` (pd.DataFrame): Cleaned data

**Returns:**
- `tuple`: (X_processed, y, preprocessor)

## Model Training Module

### `ModelTrainer` Class

#### `train_single_model(model_name, X_train, y_train, X_test, y_test)`
Train a single model with hyperparameter tuning.

**Parameters:**
- `model_name` (str): Name of the algorithm
- `X_train` (pd.DataFrame): Training features
- `y_train` (pd.Series): Training target
- `X_test` (pd.DataFrame): Test features
- `y_test` (pd.Series): Test target

**Returns:**
- `dict`: Model metrics and results

#### `train_all_models(X_train, y_train, X_test, y_test)`
Train all specified models.

**Parameters:**
- `X_train` (pd.DataFrame): Training features
- `y_train` (pd.Series): Training target
- `X_test` (pd.DataFrame): Test features
- `y_test` (pd.Series): Test target

**Returns:**
- `dict`: Results for all models

## Causal Analysis Module

### `CausalAnalyzer` Class

#### `analyze_treatment_effect(treatment, outcome, confounders)`
Analyze causal effect of a treatment.

**Parameters:**
- `treatment` (str): Treatment variable name
- `outcome` (str): Outcome variable name
- `confounders` (list): List of confounder variables

**Returns:**
- `dict`: Causal analysis results

#### `get_effect_summary()`
Get summary of all causal effects.

**Returns:**
- `pd.DataFrame`: Summary of causal effects

## Utility Functions

### `load_config(config_path)`
Load configuration from YAML file.

**Parameters:**
- `config_path` (str): Path to config file

**Returns:**
- `dict`: Configuration dictionary

### `setup_logging(logging_config_path)`
Setup logging configuration.

**Parameters:**
- `logging_config_path` (str): Path to logging config file

## Streamlit Components

### Charts Module

#### `create_roc_curve_plot(y_true, y_pred_proba)`
Create ROC curve plot.

**Parameters:**
- `y_true` (array): True labels
- `y_pred_proba` (array): Predicted probabilities

**Returns:**
- `plotly.graph_objects.Figure`: ROC curve plot

#### `create_feature_importance_plot(importance_df, top_n)`
Create feature importance plot.

**Parameters:**
- `importance_df` (pd.DataFrame): Feature importance data
- `top_n` (int): Number of top features to show

**Returns:**
- `plotly.graph_objects.Figure`: Feature importance plot

### Forms Module

#### `create_customer_input_form(config)`
Create form for customer data input.

**Parameters:**
- `config` (dict): Application configuration

**Returns:**
- `dict` or `None`: Customer data if form submitted

## Configuration Reference

### Data Configuration
```yaml
data:
  raw_file: "data/raw/dataset.csv"
  target_column: "Churn"
  test_size: 0.2
  random_state: 42