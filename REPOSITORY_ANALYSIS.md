# Repository Analysis: ChurnDashboard

## 1) High-level overview

This repository is a compact Streamlit application for exploratory analysis of customer churn data.

- **Primary app entry point:** `app.py`
- **Dependencies:** declared in `requirements.txt`
- **Usage and deployment docs:** `README.md`

The codebase is intentionally small and easy to run, with most behavior implemented in one file.

## 2) Functional behavior summary

The application provides:

1. **CSV-based ingestion**
   - Required primary dataset upload
   - Optional secondary incoming/unlabeled dataset upload
   - Optional local filesystem fallback (`data/dataset.csv` or `data/train.csv` + `data/test.csv`)

2. **Flexible churn normalization**
   - Detects likely churn columns by common names
   - Converts multiple label conventions (Yes/No, True/False, Churned/Active, numeric) to binary

3. **Filtering and segmentation**
   - Split filtering (`main`/`incoming` or `train`/`test`)
   - Category filters (subscription and payment)
   - Tenure range filter when applicable

4. **EDA outputs**
   - KPI cards (rows and churn rate)
   - Missingness chart
   - Churn driver charts (subscription, payment, tenure, usage)
   - Distribution plots and correlation with churn
   - Rule-based recommendations generated from filtered data

## 3) Architecture assessment

### Strengths

- **Fast onboarding:** minimal setup and no complex package structure.
- **User-friendly defaults:** robust column auto-detection and safe numeric conversion reduce friction.
- **Good defensive UX:** app avoids tracebacks and presents friendly messages for invalid/missing uploads.
- **Useful analytical breadth:** combines descriptive metrics, visual diagnostics, and recommendations.

### Limitations

- **Monolithic script design:** all logic is in `app.py`, mixing I/O, transformation, UI, and business logic.
- **Testing gap:** no test suite for helper functions or edge-case data handling.
- **Potential reproducibility ambiguity:** optional in-app train/test split can differ from external ML pipelines unless governed carefully.
- **Dependency pinning absent:** `requirements.txt` is unpinned, which may lead to environment drift.

## 4) Code quality and maintainability

### Positive patterns

- Helper functions (`pick_first_existing`, `safe_numeric`, `churn_to_binary`) encapsulate common operations.
- Use of `@st.cache_data` for uploaded CSV bytes is appropriate for performance.
- Data filtering logic is mostly clear and readable.

### Refactoring opportunities

1. **Modularization**
   - Move helpers and domain logic into modules (for example: `data_loading.py`, `preprocessing.py`, `insights.py`).
   - Keep `app.py` focused on Streamlit UI composition.

2. **Type clarity and validation**
   - Add stronger schema checks for required analytical columns at ingestion stage.
   - Consider explicit data contracts for labeled vs unlabeled datasets.

3. **Recommendation engine separation**
   - Extract recommendation logic into pure functions for easier testing and future expansion.

## 5) Product and UX observations

- **Strong fit for demos/interviews:** quick value with small datasets and intuitive visuals.
- **Good adaptability:** column detection supports heterogeneous source schemas.
- **Potential user confusion areas:**
  - Difference between uploaded incoming data and ML split modes may require clearer copy.
  - Correlation output can be misinterpreted as causality; brief explanatory note would help.

## 6) Risk and reliability notes

- Broad `except Exception` prevents crashes but can hide specific parsing/data issues.
- Churn mapping handles many variants, but uncommon label encodings may still silently become missing.
- Since charts are conditional on available columns, some users may see sparse dashboards without understanding why unless guidance is explicit.

## 7) Recommended next steps (prioritized)

1. **Add tests (high impact)**
   - Unit tests for `churn_to_binary`, column detection, and split logic.
2. **Modularize app logic (high impact)**
   - Separate UI from computation to improve maintainability.
3. **Pin dependency versions (medium impact)**
   - Improve deployment stability.
4. **Improve diagnostics (medium impact)**
   - Replace broad exception with targeted messages (CSV parse error, missing columns, encoding problems).
5. **Add explanatory UX text (medium impact)**
   - Clarify labeled-row behavior and correlation interpretation.

## 8) Bottom line

The repository is a well-scoped, practical Streamlit EDA dashboard with solid usability for churn exploration. Its main technical debt is structural (single-file architecture, no tests), not conceptual. A small investment in modularization and test coverage would significantly improve long-term maintainability without changing product direction.
