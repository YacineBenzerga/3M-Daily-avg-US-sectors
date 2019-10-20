import numpy as np
import pandas as pd

# Importing data on csv
data = pd.read_csv('raw_cleaned.csv', encoding='latin-1')
data = data.set_index('Symbol')
np.random.seed(0)
y = data['3M Price Returns (Daily)']
neutral_index = list(y[y == 0].index)
data = data.drop(neutral_index, axis=0)

# Explanatory and target variables
X = data.drop(['Name', 'Fixed Asset Turnover(Quartl)', 'Earning Yield(forward 1y)',
               'Total Returns Price', '3M Price Returns (Daily)'], axis=1)
y = data['3M Price Returns (Daily)']


# Mean removal and unit std
for col in X.columns:
    X[col] = X[col].apply(lambda x: (x-X[col].mean())/(X[col].std()))

# Validation Set
X_validation = X[-300:]
y_validation = y[-300:]

# Training Set
X_train = X[:-300]
y_train = y[:-300]

# Regression

# Libraries
from sklearn.model_selection import KFold, cross_val_score
kfold = KFold(10)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, explained_variance_score
from xgboost import XGBRegressor
from sklearn.linear_model import RANSACRegressor

# Modelisation

models = [('Ridge', Ridge()),
          ('Lasso', Lasso()),
          ('Elastic Net', ElasticNet()),
          ('SVR', SVR(kernel='rbf')),
          ('RF_regressor', RandomForestRegressor()),
          ('Xgboost_regressor', XGBRegressor()),
          ]

# Metric:Explained Variance
print('Explained Variance\n')
for m in models:
    this_score = cross_val_score(
        m[1], X_train, y_train, cv=kfold, scoring=make_scorer(explained_variance_score))
    print('%s Explained Variance score is %.3f +/- %.3f\n' %
          (m[0], np.mean(this_score), np.std(this_score)))


""" Explained Variance

Ridge Explained Variance score is 0.159 +/- 0.017

Lasso Explained Variance score is 0.000 +/- 0.000

Elastic Net Explained Variance score is 0.000 +/- 0.000

SVR Explained Variance score is 0.103 +/- 0.070

RF_regressor Explained Variance score is 0.084 +/- 0.066

Xgboost_regressor Explained Variance score is 0.172 +/- 0.039

 """


# Classification
from sklearn.model_selection import StratifiedKFold


y = data['3M Price Returns (Daily)']
y_validation = y[-300:]
y_train = y[:-300]
y_train_class = np.sign(y_train)
y_validation_class = np.sign(y[-300:])
cv = StratifiedKFold(n_splits=10, shuffle=True)

# Libraries
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


# Modelisation
models = [('LR', LogisticRegression()),
          ('LDA',  LDA()),
          ('QDA', QDA()),
          ('LSVC', LinearSVC()), ('RSVM', SVC(gamma=0.0001, kernel='rbf')),
          ('RF',   RandomForestClassifier()), ('AdaBoost', AdaBoostClassifier()),
          ('MLP', MLPClassifier()),
          ('Xgboost_classifier', XGBClassifier())]


print('\nAUC for different models\n')
for m in models:
    this_score = cross_val_score(
        m[1], X_train, y_train_class, cv=kfold, scoring='roc_auc')
    print('%s average AUC score is %.3f +/- %.3f' %
          (m[0], np.mean(this_score), np.std(this_score)))


# Hyperparamters Tuning through GridSearchCV(XGBoost)

param_grid_xgboost = {'n_estimators': [20, 35, 50, 100], 'max_depth': [
    3, 6, 9], 'min_child_weight': [1, 5, 15]}
classifier = XGBClassifier()
grid_xgboost = GridSearchCV(
    classifier, param_grid_xgboost, cv=5, scoring='roc_auc')
grid_xgboost.fit(X_train, y_train_class)
print('\nHyperparameter Tuning\n')
print('\nBest Score on XGboost is {} for best params {}\n'.format(
    round(grid_xgboost.best_score_, 2), grid_xgboost.best_params_))

# Hyperparamters Tuning through GridSearchCV(LinearSVC)

param_grid_lsvc = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
classifier = LinearSVC()
grid_lsvc = GridSearchCV(classifier, param_grid_lsvc, cv=5, scoring='roc_auc')
grid_lsvc.fit(X_train, y_train_class)
print('\nBest Score on LSVC is {} for best params {}\n'.format(
    round(grid_lsvc.best_score_, 2), grid_lsvc.best_params_))

# We try Bagging for LSVC estimator with best parameters
print('\nBagging With LSVC\n')
params = grid_lsvc.best_params_
Bagging = BaggingClassifier(LinearSVC(**params), n_estimators=20, n_jobs=-1)
Bagging.fit(X_train, y_train_class)
pred = Bagging.predict(X_validation)
proba = Bagging.predict_proba(X_validation)
prob_buy = [c[1] for c in proba]
prob_sell = [c[0] for c in proba]
print('\nClasses are mapped:{}\n'.format(Bagging.classes_))
print('Mean Accuracy Score', round(
    Bagging.score(X_validation, y_validation_class), 2))
print('f1-score', round(f1_score(y_validation_class, pred), 2))
print(confusion_matrix(pred, y_validation_class))
print(classification_report(
    pred, y_validation_class, target_names=['Sell', 'Buy']))

results_lsvc = pd.DataFrame({'Prediction': pred, 'Probabilties_Buy': prob_buy,
                             'Probabilities_Sell': prob_sell}, index=X_validation.index)
results_lsvc = results_lsvc.sort_values(by='Probabilties_Buy', ascending=False)
results_lsvc.to_csv('Prediction Results_lsvc.csv', index=True)

# We try Bagging for XGBoost estimator with best parameters
print('\nBagging With XGBoost\n')
params = grid_xgboost.best_params_
Bagging = BaggingClassifier(XGBClassifier(
    **params), n_estimators=20, n_jobs=-1)
Bagging.fit(X_train, y_train_class)
pred = Bagging.predict(X_validation)
proba = Bagging.predict_proba(X_validation)
prob_buy = [c[1] for c in proba]
prob_sell = [c[0] for c in proba]
print('\nClasses are mapped:{}\n'.format(Bagging.classes_))
print('Mean Accuracy score', round(
    Bagging.score(X_validation, y_validation_class), 2))
print('f1-score', round(f1_score(y_validation_class, pred), 2))
print(confusion_matrix(y_validation_class, pred))
print(classification_report(y_validation_class,
                            pred, target_names=['Sell', 'Buy']))

results_xgboost = pd.DataFrame({'Prediction': pred, 'Probabilties_Buy': prob_buy,
                                'Probabilities_Sell': prob_sell}, index=X_validation.index)
results_xgboost = results_xgboost.sort_values(
    by='Probabilties_Buy', ascending=False)
results_xgboost.to_csv('Prediction Results_xgboost.csv', index=True)


""" AUC for different models

LR average AUC score is 0.698 +/- 0.028
LDA average AUC score is 0.699 +/- 0.028
QDA average AUC score is 0.677 +/- 0.040
LSVC average AUC score is 0.698 +/- 0.028
RSVM average AUC score is 0.616 +/- 0.035
RF average AUC score is 0.636 +/- 0.034
AdaBoost average AUC score is 0.685 +/- 0.030
MLP average AUC score is 0.697 +/- 0.035
Xgboost_classifier average AUC score is 0.685 +/- 0.026

Hyperparameter Tuning


Best Score on XGboost is 0.69 for best params {'max_depth': 3, 'min_child_weight': 15, 'n_estimators': 35}


Best Score on LSVC is 0.7 for best params {'C': 0.001}


Bagging With LSVC


Classes are mapped:[-1.  1.]

Mean Accuracy Score 0.77
f1-score 0.86
[[ 12   5]
 [ 64 219]]
             precision    recall  f1-score   support

       Sell       0.16      0.71      0.26        17
        Buy       0.98      0.77      0.86       283

avg / total       0.93      0.77      0.83       300


Bagging With XGBoost


Classes are mapped:[-1.  1.]

Mean Accuracy score 0.74
f1-score 0.84
[[ 20  56]
 [ 23 201]]
             precision    recall  f1-score   support

       Sell       0.47      0.26      0.34        76
        Buy       0.78      0.90      0.84       224

avg / total       0.70      0.74      0.71       300 """
