# IDX-Exchange-DS-Project

### Project Objective
The goal of this project is to build and train a machine learning model to predict the `ClosePrice` (final sales price) of single residential properties in California using historical real estate data sourced from the California Regional Multiple Listing Service (CRMLS).

---

### Weekly Milestones & Progress

#### Week 1: Orientation & Setup
* Analyzed the core assignment objectives and project scope.
* Retrieved a minimum of 6 months of historical CRMLSSold CSV data files.
* Examined the MetaData documentation to identify critical target features and column specifications.

#### Week 2: Data Exploration
* Imported the 6-month real estate dataset into a Jupyter Notebook via pandas.
* Mapped out and analyzed distributions for the five core metrics: `ClosePrice`, `LivingArea`, `Bedrooms`, `Bathrooms`, and `LotSize`.
* Filtered the scope strictly to `PropertyType = "Residential"` and `PropertySubType = "Single Family Residence"` rows.
* **Deliverable:** `01_exploration.ipynb` (Initial Jupyter Notebook detailing the exploratory data analysis plots).

#### Week 3: Data Preprocessing
* Implemented clean data preprocessing routines by dropping listings completely missing critical values (`ClosePrice`, `LivingArea`) and applying median imputation to resolve missing values (`BedroomsTotal`, `BathroomsTotalInteger`, `LotSizeSquareFeet`).
* Normalized numerical features (`LivingArea`, `LotSizeSquareFeet`, etc.) using `StandardScaler` to ensure a balanced, standardized feature set.
* Built a dynamic, rolling time-series train/test split, isolating the most recent month (May 2026) as the testing set and defining a tunable sliding historical window (6 months) preceding it for model training.
* **Deliverables:**
  * `02_preprocessing.ipynb` (Clean preprocessing pipeline and dynamic train/test splitting notebook).
  * `cleaned_sales_data.csv` (Fully cleaned, normalized, and imputed master dataset exported for training).

#### Week 4: Baseline Model
* Evaluated predictions on the May 2026 test set using the Coefficient of Determination ($R^2$) metric to establish an initial performance benchmark.
* Documented training constraints, hyperparameter baselines (using a default 6-month historical window), and initial prediction variance metrics.
* **Deliverable:** `03_baseline_model.ipynb` (Jupyter Notebook demonstrating baseline model setup, evaluation run, and recorded $R^2$ performance).

#### Week 5: Outlier Mitigation & Model Expansion
* Applied price boundary filtering (\$100,000 to \$5,000,000) to strip out recording anomalies and extreme luxury edge cases that were skewing variance.
* Implemented and compared multiple baseline regression architectures, evaluating Linear Regression, Decision Trees, and Random Forest Regressors on the standardized feature set.
* **Deliverable:** `04_model_expansion.ipynb` (Model comparison notebook establishing post-filtering benchmarks).

#### Week 6: Feature Engineering & Spatial Integration
* Engineered domain-specific features including `bed_bath_ratio` and `property_age` relative to transaction dates.
* Integrated California Department of Education GIS boundary shapefiles via `GeoPandas` to execute spatial joins on property coordinates.
* Calculated `school_district_avg_price` as a high-signal geographic feature, which drove Random Forest $R^2$ accuracy from 0.37 to 0.68.
* **Deliverables:**
  * `05_feature_engineering.ipynb` (Notebook demonstrating spatial join and feature creation).
  * California School District shapefiles integrated into `data/school_districts/`.

#### Week 7: Gradient Boosting & Hyperparameter Tuning
* Upgraded model architecture to `XGBoost` to handle non-linear tabular interactions via gradient boosted decision trees.
* Performed hyperparameter optimization with `RandomizedSearchCV` across 15 parameter combinations, tuning `learning_rate`, `max_depth`, `subsample`, and regularization parameters (`reg_alpha`, `reg_lambda`).
* Extracted feature importances, confirming that spatial school district averages accounted for 32.4% of total predictive power, followed by bathroom count (28.7%) and living area (18.4%).
* **Deliverable:** `06_model_optimization.ipynb` (Hyperparameter tuning and feature importance notebook achieving an out-of-sample $R^2$ of 0.6905).

#### Week 8: Model Evaluation & Error Diagnostics
* Evaluated out-of-sample performance on the May 2026 test set beyond $R^2$, computing Mean Absolute Error (MAE: \$284,128.99), Root Mean Squared Error (RMSE: \$460,496.79), Mean Absolute Percentage Error (MAPE: 24.70%), and Median Absolute Percentage Error (MdAPE: 17.66%).
* Segmented residual errors by market tiers (Entry Level, Mid Tier, Upper Tier, Luxury), identifying that the model achieved peak accuracy on core Mid-Tier homes (\$500k–\$1M) with a median percentage error of 15.40%.
* **Deliverables:**
  * `06_evaluation.ipynb` (Comprehensive residual diagnostic and evaluation notebook).
  * `metrics_summary.csv` (Exported breakdown of performance metrics and error distributions across market tiers).

---

### Instructions to Re-Run the Pipeline

1. **Clone the Repository & Set Up Environment:**
   ```bash
   git clone [https://github.com/aaron-reimer/IDX-Exchange.git](https://github.com/aaron-reimer/IDX-Exchange.git)
   cd IDX-Exchange
   python3 -m venv venv
   source venv/bin/activate
   pip install pandas numpy geopandas shapely scikit-learn xgboost jupyter
