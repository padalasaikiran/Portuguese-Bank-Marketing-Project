# Portuguese Bank Marketing Project

Predicts which customers will subscribe to a term deposit after a phone call, so a bank's marketing team can call the most promising customers first.

## Project overview
- Data: 41,188 phone calls from a Portuguese bank (2008 to 2010), 20 features. 41,176 rows remain after removing 12 duplicates.
- Task: binary classification (subscribed: yes or no). Only 11.3% of customers subscribed.
- Final model: tuned Random Forest (F1 0.508, ROC-AUC 0.804, recall 0.613, precision 0.433).

## Files in this repository
| File | Description |
|---|---|
| Portuguese_Bank_Term_Deposit_Prediction (4).ipynb | Full analysis: data overview, wrangling, 14 charts, 4 hypothesis tests, feature engineering, 5 models, model comparison, feature importance, suggestions, conclusion and challenges report |
| bank-additional-full.csv | Dataset (semicolon separated) |
| best_model_random_forest.pkl | Saved final model with its scaler and column order |
| requirements.txt | Python packages |

## Method in short
1. Cleaned hidden "unknown" values and the special code pdays = 999.
2. Dropped call duration, because it is only known after the call (data leakage).
3. Capped outliers (IQR), encoded categories, log-transformed campaign, stratified 80/20 split, scaled features.
4. Balanced the training data with SMOTE (test set untouched).
5. Tuned Logistic Regression, Decision Tree, Random Forest, Gradient Boosting and XGBoost with 3-fold cross-validated grid search (SMOTE inside the pipeline).
6. Compared ten model versions on Precision, Recall, F1 and ROC-AUC.

## Key findings
- Economic indicators (Euribor rate, employment) make up about 53% of feature importance.
- Previous campaign success, cellular contact and a low interest-rate period go with higher subscription.
- Calling the 1,313 customers the model flags reaches 61% of all subscribers, and about 43% of them subscribe, against 11.3% for random calling.

## How to run
1. Clone the repository: git clone https://github.com/padalasaikiran/Portuguese-Bank-Marketing-Project.git
2. Install packages: pip install -r requirements.txt
3. Open the notebook in Jupyter or Google Colab. Keep bank-additional-full.csv in the same folder as the notebook.
4. Run all cells (the tuning cells take several minutes).

## Using the saved model
The pickle file holds a dictionary with the model, the scaler, the feature column order and the numeric columns. New customers must be prepared exactly like the training data (same encoding, capping and log transform) before scaling and prediction.

## Limitations
The top models perform similarly, the data covers one period (2008 to 2010), and the results are associations, not causes.
