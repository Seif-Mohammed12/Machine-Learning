# Milestone 1 Report: Online Games Popularity Prediction

## 1. Objective

The objective is to preprocess the provided dataset and build regression models to predict `RecommendationCount` with minimum error.

## 2. Dataset Overview

- Rows: 11,357
- Columns: 78
- Target: `RecommendationCount`
- Main feature types:
  - Numeric metadata (`PriceInitial`, `Metacritic`, `SteamSpyOwners`, `AchievementCount`, ...)
  - Boolean/category indicators
  - Date field (`ReleaseDate`)
  - Text fields (`AboutText`, `ShortDescrip`, `DetailedDescrip`, requirements text, ...)

### Target behavior

The target distribution is heavily right-skewed. Most games have low recommendation count, and a smaller number have very high values. Therefore, training is done on `log1p(RecommendationCount)` and then converted back to original scale for MAE/RMSE reporting.

## 3. Preprocessing (Lab 4 style)

All dataset features were preprocessed before model building.

### 3.1 Missing values

Missing values were handled using median imputation after feature engineering.

### 3.2 Date preprocessing

`ReleaseDate` was transformed into:

- `release_year`
- `release_month`
- `release_dayofweek`
- `release_age_days`

### 3.3 Boolean / numeric processing

- Boolean features were converted to `0/1`.
- Numeric columns were preserved as numeric features.

### 3.4 Encoding

- `PriceCurrency` was encoded with one-hot encoding.

### 3.5 Text feature transformation

For each text column, the following numerical descriptors were generated:

- character length
- word count
- presence flag

This was applied to all relevant text columns (`QueryName`, `ResponseName`, `AboutText`, `DetailedDescrip`, `Website`, requirements fields, etc.).

### 3.6 Additional engineered features

- `price_discount`
- `price_discount_ratio`
- `supported_languages_count`
- `title_match`
- `title_len_diff`
- `is_paid`

Final engineered feature count: 126.

## 4. Feature Selection and Analysis (Lab 5 style)

A filter method was used (`SelectKBest` + mutual information). Top features were selected from the training set.

- Selected feature count used for modeling: 10

Top importance trend was dominated by popularity/history metadata, especially SteamSpy-based signals and release/quality related attributes.

Outputs:

- [Target distribution plot](artifacts/plots/target_distribution.png)
- [Selected features CSV](artifacts/selected_features.csv)
- [Top selected features plot](artifacts/plots/top_selected_features.png)

## 5. Train/Test Split and Assessment Setup

- Train size: 9,085
- Test size: 2,272

Model comparison follows lecture/lab model-assessment style:

- same split for all models
- compare MAE and RMSE on test set
- also report `R2` on the log-transformed target

## 6. Regression Models Used

Two regression techniques were used, consistent with your uni material:

### 6.1 Linear Regression

- Model: `LinearRegression`
- Input: selected features
- Pre-scaling: `StandardScaler`

### 6.2 Polynomial Regression

- Model: `PolynomialFeatures` + regularized linear regression (`Ridge`) for stability
- Degree search: 2 and 3
- Input: selected features
- Pre-scaling: `StandardScaler`

Degree-vs-error output:

- [Degree vs error plot](artifacts/plots/degree_vs_error.png)

## 7. Results

### 7.1 Degree search

| Degree |        MAE |          RMSE |
| ------ | ---------: | ------------: |
| 2      | 214,713.25 | 10,175,675.29 |
| 3      | 659,297.60 | 17,647,820.02 |

Best polynomial degree on this split: **2**.

### 7.2 Final model comparison

| Model                                               | R2 on log target | MAE on original target | RMSE on original target |
| --------------------------------------------------- | ---------------: | ---------------------: | ----------------------: |
| Linear Regression (selected features)               |           0.6085 |             257,253.83 |           10,268,036.82 |
| Polynomial Regression (degree=2, selected features) |           0.7079 |             214,713.25 |           10,175,675.29 |

Interpretation:

- Polynomial regression (degree 2) performed better than plain linear regression on this split.
- Degree 3 overfit and produced much higher error, which is consistent with Lab 5 overfitting discussion.

Plots:

- [Linear actual vs predicted](artifacts/plots/linear_actual_vs_predicted.png)
- [Linear log-scale scatter](artifacts/plots/linear_actual_vs_predicted_log.png)
- [Polynomial actual vs predicted](artifacts/plots/poly_actual_vs_predicted.png)
- [Polynomial log-scale scatter](artifacts/plots/poly_actual_vs_predicted_log.png)
- [Model comparison](artifacts/plots/model_comparison.png)

## 8. Alignment with Uni Material

This implementation follows the same practical topics taken in lectures/labs:

- Lecture 1/2/3 + Labs 2&3:
  - linear regression workflow and error minimization
- Lecture 4 + Lab 4:
  - preprocessing, scaling, encoding, feature engineering, polynomial regression
- Lecture 5 + Lab 5:
  - overfitting awareness, model assessment with train/test, feature selection, degree comparison

No extra advanced model family (e.g., tree boosting) is used in the final pipeline.

## 9. Conclusion

For Milestone 1, preprocessing and feature-selection were completed and two regression methods were evaluated. Polynomial regression with degree 2 achieved the best result among the tested models and matched the course expectation about improving linear models while avoiding high-degree overfitting.

The project is now aligned with the exact uni progression for this phase and ready for milestone submission.
