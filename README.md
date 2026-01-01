# Customer Churn EDA Dashboard

An interactive **Streamlit dashboard** for **exploratory data analysis (EDA)** of customer churn datasets.
Designed for **data storytelling, interviews, and ML readiness**, this app works with a **single dataset** by default and optionally supports **train/test splits** for modeling workflows.

---

## Live Demo

Once deployed on Streamlit Community Cloud, the app will be accessible via a public URL.
`https://churndashboard-sbwkl8hexrffnrltucgsty.streamlit.app/`
---

## Features

* Single-file-first data loading (recommended)
* Optional train/test split for ML workflows
* Automatic churn column detection (robust to naming)
* Interactive filters (subscription, payment method, tenure)
* Key KPIs (churn rate, dataset size)
* Churn drivers analysis

  * Subscription / contract impact
  * Payment method impact
  * Tenure vs churn
  * Usage vs churn
* Data quality checks (missing values)
* Correlation analysis with churn
* Actionable business recommendations
* Streamlit Cloud–ready (no local paths required)

---

## Dataset Requirements

### Minimum Required Column

* **Churn** (target variable)

Accepted formats:

* Numeric: `0 / 1`
* Text: `Yes / No`, `Churned / Active`, `True / False`

### Commonly Detected Columns (Optional)

| Category     | Examples                                                                |
| ------------ | ----------------------------------------------------------------------- |
| Subscription | `SubscriptionType`, `Plan`, `Contract`                                  |
| Payment      | `PaymentMethod`                                                         |
| Tenure       | `Tenure`, `AccountAge`, `MonthsAsCustomer`                              |
| Usage        | `MonthlyCharges`, `ViewingHoursPerWeek`, `SupportTicketsPerMonth`, etc. |

The app **auto-detects** available columns and adapts automatically.

---

## How the App Loads Data

### Option 1 (Recommended): Single CSV Upload

Upload **one dataset** containing all customers.

The app will:

* Treat all rows as one dataset
* Compute churn insights directly
* Optionally split data for ML (see below)

### Option 2: Optional ML-Ready Train/Test Split

You can enable an **optional split** from the sidebar:

* Choose split ratio (e.g., 80/20)
* Random seed for reproducibility
* Useful for:

  * Model validation
  * Feature comparison
  * Leakage-free analysis

### Option 3: Train/Test Files (Advanced)

Upload:

* `train.csv`
* `test.csv`

The app will tag rows internally and allow comparison.

---

## Running Locally

### 1. Clone the Repository

```bash
git clone https://github.com/jaybaragadi/ChurnDashboard.git
cd ChurnDashboard
```

### 2. Create Virtual Environment (Optional)

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run app.py
```

---

## Deploying on Streamlit Community Cloud

1. Push code to GitHub
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Click **New app**
4. Select:

   * Repository: `jaybaragadi/ChurnDashboard`
   * Branch: `main`
   * Main file: `app.py`
5. Click **Deploy**

No dataset is required in the repo — users upload files via UI.

---

## Project Structure

```text
ChurnDashboard/
│
├── app.py              # Streamlit application
├── requirements.txt    # Dependencies
├── README.md           # Project documentation
```

---


## Tech Stack

* Python
* Streamlit
* Pandas
* NumPy
* Plotly

---

## License

This project is open-source and intended for **learning, portfolios, and demos**.

---

If you want, next I can help you with:

* Final **Streamlit Cloud polish**
* **ML model integration** (Logistic / XGBoost)
* **Interview explanation script**
* **Portfolio write-up (Medium / LinkedIn)**
