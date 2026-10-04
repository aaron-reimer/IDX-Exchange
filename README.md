# IDX-Exchange-DS-Project

### Project Objective
The goal of this project is to build and train a machine learning model to predict the `ClosePrice` (final sales price) of single-family residential properties in California using historical real estate data sourced from the California Regional Multiple Listing Service (CRMLS).

## Results (test month: May 2026)
- Model: XGBoost, tuned with RandomizedSearchCV (15 candidates, 3-fold cross-validation)
- Test R²: 0.69
- Median absolute percentage error: 17.7% overall, 15.4% on mid-tier homes (\$500K-\$1M)
- Data: 258,669 raw CRMLS records, 130,220 single-family sales; trained on 58,295 sales (Nov 2025 - Apr 2026) and tested on 11,811 sales (May 2026)
- Key finding: a school-district average price feature (built from a GIS spatial join) was the top predictor at 32.4% of feature importance and raised Random Forest R² from 0.37 to 0.68. See Known Limitations below.

**Data:** CRMLS sold-listing data is proprietary, so data files are not included in this repo. The notebooks expect twelve monthly files (`CRMLSSold202506.csv` through `CRMLSSold202605.csv`) in `data/` and the California Department of Education school district shapefile in `data/school_districts/`. Notebooks check `data/` first, then `../data/`.

**Presentation:** [IDX Exchange final presentation](IDX%20Exchange%20-%20ds56%20Presentation.pdf)

## Known Limitations
- **Feature leakage:** `school_district_avg_price` is the mean `ClosePrice` per school district computed over all 12 months of data, including the May 2026 test month. The feature therefore contains information about the target for test homes, so the test R² and this feature's importance are likely optimistic. A stricter version would compute district averages from the training months only.
- **Full-dataset preprocessing:** Median imputation values and `StandardScaler` parameters were also computed on the full dataset rather than the training window alone.
- **Notebook 07:** It evaluates a tuned model named `final_model` that is not defined in the notebook. See step 4 of the re-run instructions.

---

### Weekly Milestones & Progress

#### Week 1: Orientation & Setup
* Analyzed the core assignment objectives and project scope.
* Retrieved 12 monthly CRMLS sold-listing CSV files (June 2025 - May 2026).
* Examined the MetaData documentation to identify critical target features and column specifications.

#### Week 2: Data Exploration
* Combined the 12 monthly files with pandas into one dataset of 258,669 records and 78 columns.
* Filtered to `PropertyType = "Residential"` and `PropertySubType = "SingleFamilyResidence"`, leaving 130,220 records.
* Summarized the five core metrics: `ClosePrice`, `LivingArea`, `BedroomsTotal`, `BathroomsTotalInteger`, and `LotSizeSquareFeet`.
* Set plotting boundaries (price between \$10K and \$6.5M, living area under 10,000 sq ft, lot size under 150,000 sq ft) and plotted bedroom and bathroom count distributions.
* **Deliverable:** `01_exploration.ipynb`

#### Week 3: Data Preprocessing
* Dropped listings missing `ClosePrice` or `LivingArea` (130,152 records remaining) and filled missing `BedroomsTotal`, `BathroomsTotalInteger`, and `LotSizeSquareFeet` values with the median.
* Standardized `LivingArea`, `BedroomsTotal`, `BathroomsTotalInteger`, and `LotSizeSquareFeet` with `StandardScaler`.
* Built a time-based train/test split: May 2026 as the test set (12,017 sales) and the 6 months before it (Nov 2025 - Apr 2026) as the training set (59,415 sales).
* Exported the cleaned dataset to `cleaned_sales_data.csv` (not included in this repo).
* **Deliverable:** `02_preprocessing.ipynb`

#### Week 4: Baseline Model
* Trained a Linear Regression baseline on the four standardized features using the 6-month training window.
* Baseline test R² on the May 2026 test set: 0.2746.
* **Deliverable:** `03_baseline_model.ipynb`

#### Week 5: Outlier Mitigation & Model Comparison
* Removed sales priced below \$100,000 or above \$5,000,000, leaving 58,295 training and 11,811 test sales.
* Compared models on the same four features: Linear Regression (R² 0.3225), Decision Tree with max depth 10 (R² 0.3239), and Random Forest with 100 trees and max depth 12 (R² 0.3703).
* **Deliverable:** `04_model_comparison.ipynb`

#### Week 6: Feature Engineering & Spatial Integration
* Engineered `bed_bath_ratio` and `property_age` (2026 minus `YearBuilt`, limited to 0-150 years).
* Joined California Department of Education school district boundaries to property coordinates with `GeoPandas` spatial joins, then computed `school_district_avg_price`, the mean `ClosePrice` per district over the full dataset (see Known Limitations).
* Results with the new features: Linear Regression R² 0.6015 and Random Forest R² 0.6758 (up from 0.3703).
* **Deliverable:** `05_feature_engineering.ipynb`

#### Week 7: Gradient Boosting & Hyperparameter Tuning
* Trained a default XGBoost model (R² 0.6853), then tuned it with `RandomizedSearchCV` (15 candidates, 3-fold cross-validation) over `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_alpha`, and `reg_lambda`.
* Best parameters: `n_estimators=300`, `max_depth=10`, `learning_rate=0.05`, `subsample=0.7`, `colsample_bytree=0.7`, `reg_alpha=1.0`, `reg_lambda=10.0`. Tuned XGBoost test R²: 0.6905.
* Feature importance: `school_district_avg_price` (32.4%), `BathroomsTotalInteger` (28.7%), `LivingArea` (18.4%), `bed_bath_ratio` (6.9%), `property_age` (5.9%), `LotSizeSquareFeet` (4.2%), `BedroomsTotal` (3.5%).
* **Deliverable:** `06_model_optimization.ipynb`

#### Week 8: Model Evaluation & Error Diagnostics
* Evaluated the tuned model on the May 2026 test set (11,811 sales): R² 0.6905, MAE \$284,128.99, RMSE \$460,496.79, MAPE 24.70%, and MdAPE 17.66%.
* Segmented errors by price tier; accuracy was best on mid-tier homes (15.40% median error):

| Price tier | Test sales | Median % error | Mean % error | Mean absolute error |
|---|---|---|---|---|
| Entry Level (under \$500K) | 1,681 | 21.11% | 37.78% | \$129,361 |
| Mid Tier (\$500K-\$1M) | 4,897 | 15.40% | 22.46% | \$166,904 |
| Upper Tier (\$1M-\$2M) | 3,750 | 16.88% | 21.13% | \$299,163 |
| Luxury (\$2M-\$5M) | 1,483 | 24.06% | 26.33% | \$808,633 |

* Exported the tier summary to `data/metrics_summary.csv` (not included in this repo).
* **Deliverable:** `07_model_evaluation.ipynb`

---

### Instructions to Re-Run the Pipeline

1. Clone the repository and set up the environment:

```bash
git clone https://github.com/aaron-reimer/IDX-Exchange.git
cd IDX-Exchange
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy geopandas shapely scikit-learn xgboost jupyter matplotlib seaborn
```

2. Add the data files described in the Data note above.
3. Run `jupyter notebook` and execute the notebooks in numerical order. Notebook 02 creates `cleaned_sales_data.csv`, which notebooks 03-07 read.
4. Before running the metrics cell in `07_model_evaluation.ipynb`, define the tuned model it evaluates:

```python
final_model = XGBRegressor(subsample=0.7, reg_lambda=10.0, reg_alpha=1.0, n_estimators=300,
                           max_depth=10, learning_rate=0.05, colsample_bytree=0.7,
                           random_state=42, n_jobs=-1)
final_model.fit(X_train, y_train)
```
