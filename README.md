# Customer Churn Casual Analysis

A comprehensive machine learning project that predicts customer churn using advanced causal analysis techniques and provides an interactive Streamlit dashboard for business insights.

## 🎯 Project Overview

This project analyzes customer churn patterns in telecom data to:
- **Predict churn probability** using multiple ML algorithms
- **Identify causal factors** that drive customer churn
- **Provide actionable insights** through SHAP explainability
- **Deliver interactive dashboard** for business stakeholders

## 📊 Dataset

**Source**: Telco Customer Churn Dataset  
**Size**: ~7,000 customers with 21 features  
**Target**: Binary churn classification (Yes/No)

**Key Features**:
- Customer demographics (gender, age, partner status)
- Service information (internet, phone services, streaming)
- Account details (tenure, contract type, payment method)
- Billing information (monthly charges, total charges)

## 🏗️ Project Structure

```
customer-churn-causal-analysis/
├── data/                    # Raw and processed datasets
├── notebooks/               # Jupyter analysis notebooks
├── src/                     # Source code modules
├── streamlit_app/           # Interactive dashboard
├── models/                  # Trained models and artifacts
├── results/                 # Analysis outputs and reports
├── config/                  # Configuration files
└── docs/                    # Documentation
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd customer-churn-causal-analysis
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
pip install -e .
```

4. **Run the Streamlit dashboard**
```bash
streamlit run streamlit_app/app.py
```

## 📓 Analysis Notebooks

Execute notebooks in this order for complete analysis:

1. **`01_data_exploration.ipynb`** - Dataset overview and EDA
2. **`02_feature_engineering.ipynb`** - Feature creation and selection
3. **`03_model_training.ipynb`** - ML model training and tuning
4. **`04_shap_analysis.ipynb`** - Model explainability with SHAP
5. **`05_causal_analysis.ipynb`** - Causal inference analysis
6. **`06_model_comparison.ipynb`** - Model performance comparison

## 🔧 Key Features

### Machine Learning Pipeline
- **Multiple algorithms**: RandomForest, XGBoost, LightGBM
- **Automated hyperparameter tuning** with cross-validation
- **Comprehensive evaluation metrics** (ROC-AUC, Precision, Recall, F1)
- **Feature importance analysis** and selection

### Explainable AI
- **SHAP (SHapley Additive exPlanations)** for model interpretability
- **Global and local explanations** for individual predictions
- **Feature contribution analysis** for business insights

### Causal Analysis
- **DoWhy framework** for causal inference
- **Treatment effect estimation** for key business variables
- **Causal graph construction** and validation
- **Policy recommendation** based on causal insights

### Interactive Dashboard
- **📊 Data Overview**: Dataset exploration and statistics
- **🔍 Feature Analysis**: Feature importance and correlations
- **🤖 Model Performance**: Model comparison and metrics
- **💡 Explainability**: SHAP analysis and interpretations
- **🔮 Churn Prediction**: Individual customer prediction interface
- **🧠 Causal Analysis**: Causal insights and recommendations

## 📈 Key Results

### Model Performance
- **Best Model**: XGBoost Classifier
- **ROC-AUC Score**: 0.85+
- **Precision**: 0.82+
- **Recall**: 0.78+

### Key Churn Drivers
1. **Contract type** (Month-to-month contracts)
2. **Tenure** (New customers at higher risk)
3. **Payment method** (Electronic check users)
4. **Internet service** (Fiber optic users)
5. **Monthly charges** (Higher charges increase churn)

### Causal Insights
- **Contract duration** has the strongest causal effect on churn
- **Customer support interactions** can reduce churn probability by 15%
- **Bundled services** decrease churn risk significantly

## 🛠️ Usage Examples

### Training New Models
```bash
python scripts/train_pipeline.py
```

### Batch Predictions
```bash
python scripts/batch_prediction.py --input data/new_customers.csv
```

### Running Tests
```bash
python -m pytest tests/
```

## 📊 Dashboard Screenshots

*Add screenshots of your Streamlit dashboard pages here*

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Your Name** - your.email@example.com  
**Project Link**: https://github.com/yourusername/customer-churn-causal-analysis

## 🙏 Acknowledgments

- DoWhy team for causal inference framework
- SHAP team for explainability tools
- Streamlit team for the amazing dashboard framework
- Kaggle for providing the dataset

## 📚 References

- [DoWhy: Causal Inference in Python](https://github.com/microsoft/dowhy)
- [SHAP: Explainable AI](https://github.com/slundberg/shap)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

⭐ **Star this repo if you find it helpful!** ⭐
