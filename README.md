# Customer Churn EDA Dashboard

An interactive **Streamlit dashboard** for exploratory data analysis (EDA) of customer churn datasets.
The application enables business-focused insights into churn drivers such as subscription type, payment method, tenure, and usage behavior.

The dashboard is designed to work seamlessly on **Streamlit Community Cloud** using **file upload**, with optional support for local datasets during development.

---

## Live Demo

Deployed on Streamlit Community Cloud:

```
https://churndashboard-<your-app-id>.streamlit.app
```

---

## Features

* Interactive KPI summary (row counts, churn rate)
* Churn analysis by:

  * Subscription / contract type
  * Payment method
  * Customer tenure
  * Usage and engagement metrics
* Dynamic filtering (dataset split, tenure range, categories)
* Missing-value analysis
* Distribution and correlation analysis
* Business-oriented recommendations based on EDA
* Safe file-upload flow (no runtime crashes if data is missing)

---

## Tech Stack

* Python 3.10+
* Streamlit
* Pandas
* NumPy
* Plotly

---

## Project Structure

```
ChurnDashboard/
│
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
│
└── data/                # Optional local datasets (not required for Cloud)
    ├── train.csv
    └── test.csv
```

---

## Dataset Requirements

The dashboard expects **two CSV files**:

* `train.csv`
* `test.csv`

### Required / Supported Columns

The app auto-detects columns. Common supported names include:

**Target**

* `Churn`, `Exited`, `Target`, `label`

**Customer attributes**

* `SubscriptionType`, `Plan`, `Contract`
* `PaymentMethod`
* `AccountAge`, `Tenure`, `MonthsAsCustomer`

**Usage metrics (optional)**

* `ViewingHoursPerWeek`
* `AverageViewingDuration`
* `ContentDownloadsPerMonth`
* `MonthlyCharges`
* `TotalCharges`
* `SupportTicketsPerMonth`
* `WatchlistSize`

Column names are flexible; the app adapts automatically.

---

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/jaybaragadi/ChurnDashboard.git
cd ChurnDashboard
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

### 5. Provide data

* Upload `train.csv` and `test.csv` via the UI
  **OR**
* Place them in:

  ```
  data/train.csv
  data/test.csv
  ```

---

## Deploying on Streamlit Community Cloud

1. Push the repository to GitHub
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click **New app**
5. Select:

   * Repository: `ChurnDashboard`
   * Branch: `main`
   * Main file path: `app.py`
6. Click **Deploy**

The app will automatically redeploy on every GitHub push.

---

## Design Decisions

* **File upload first**: Ensures compatibility with Streamlit Cloud where local files are not guaranteed
* **No hard crashes**: Graceful prompts guide users to upload data
* **Auto column detection**: Works across multiple churn datasets
* **Business-oriented storytelling**: Focus on insights and actions, not just charts

---

## Future Enhancements

* Feature importance using ML models
* Cohort-based churn analysis
* Model training and prediction integration
* Downloadable insight reports
* Authentication and multi-dataset support

---

## Author

**Jaya Sheela Baragadi**
GitHub: [https://github.com/jaybaragadi](https://github.com/jaybaragadi)
LinkedIn: [https://linkedin.com/in/jayasheela-baragadi](https://linkedin.com/in/jayasheela-baragadi)


* Create a **short README for recruiters**
* Review `requirements.txt` for Streamlit Cloud compatibility
