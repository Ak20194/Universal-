# 🏦 Universal Bank — Personal Loan Propensity Dashboard

A production-grade Streamlit dashboard for Universal Bank's marketing team to predict and analyse personal loan acceptance using machine learning.

---

## 🚀 Live Demo
Deploy on [Streamlit Community Cloud](https://streamlit.io/cloud) — free hosting for public GitHub repos.

---

## 📦 Project Structure

```
universal_bank_app/
├── app.py                  # Main Streamlit application
├── UniversalBank.csv       # Training dataset (5,000 customers)
├── test_data_sample.csv    # Sample test file for predictions (100 rows, no target)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 📊 Dashboard Features

### Tab 1 — Overview & EDA
- KPI cards: Total customers, acceptance rate, avg income, CC spend
- Target variable distribution (pie chart)
- Income distribution by loan status
- Acceptance rate by education level & family size
- Correlation heatmap across all features
- Credit card spend violin plot

### Tab 2 — Deep Dive Analysis
- CD & Securities account impact on loan conversion
- Digital banking behaviour analysis
- Income × Education heatmap (prescriptive targeting)
- Mortgage holder analysis
- Age group profiling with dual-axis chart
- Prescriptive action plan panel

### Tab 3 — ML Models
- Performance table: Train/Test Accuracy, Precision, Recall, F1, ROC-AUC
- Single ROC curve comparing all 3 models
- Feature importance chart (best model)
- Confusion matrices with counts & percentages for all models

### Tab 4 — Predict New Customers
- Upload any CSV (same schema, no target column)
- Choose model + probability threshold
- Scored output with Propensity Segment (Low/Medium/High/Very High)
- Download full results CSV

---

## 🤖 Models Used
| Model | Notes |
|-------|-------|
| Decision Tree | Interpretable baseline |
| Random Forest | Ensemble, handles imbalance well |
| Gradient Boosted Tree | Highest AUC, best for ranking |

**Class imbalance** handled with **SMOTE** (Synthetic Minority Over-sampling Technique).

---

## ⚙️ Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/universal-bank-dashboard.git
cd universal-bank-dashboard

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub (make sure `UniversalBank.csv` is included)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set `app.py` as the main file
4. Click **Deploy** — done!

---

## 📁 Test Data Format

The prediction tab accepts a CSV with these columns (no `Personal Loan` column):

```
ID, Age, Experience, Income, ZIP Code, Family, CCAvg, Education,
Mortgage, Securities Account, CD Account, Online, CreditCard
```

Download `test_data_sample.csv` from the app for a ready-to-use example.

---

## 📌 Column Reference

| Column | Description |
|--------|-------------|
| ID | Customer ID |
| Age | Age in years |
| Experience | Years of professional experience |
| Income | Annual income ($000s) |
| ZIP Code | Home zip code |
| Family | Family size |
| CCAvg | Avg monthly credit card spend ($000s) |
| Education | 1=Undergrad, 2=Graduate, 3=Advanced/Professional |
| Mortgage | Mortgage value ($000s) |
| Personal Loan | Target: 1=Accepted, 0=Not accepted |
| Securities Account | 1=Yes, 0=No |
| CD Account | 1=Yes, 0=No |
| Online | Uses internet banking: 1=Yes, 0=No |
| CreditCard | Uses bank credit card: 1=Yes, 0=No |

---

*Built for Universal Bank Marketing Analytics — 2024*
