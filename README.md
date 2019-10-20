Predicting-3M-daily-AVG-US-sectors
Introduction:

The efficient-market hypothesis posits that stock prices are a function of information and rational expectations, and that newly revealed information about a company's prospects is almost immediately reflected in the current stock price. This would imply that all publicly known information about a company, which obviously includes its price history, would already be reflected in the current price of the stock, in this regard we try to understand the drivers of price rate return, in particular the 3 month daily average return of a stock.

Data:

Input data:

The data used in this project consist of several risk factors

Price to Book Value , Gross Profit(Quarterly) , Return on Equity (TTM) , Free Cash Flow Yield(TTM) , 30 Day Average Daily Volume , Momentum Score , Market Cap .

Target:

3M Price Returns (Daily)

for a total of 3166 symbols,randomly generated from a list of 7 US sectors (ETFs, Industrials, Financials, Energy, Technology, Materials,Healthcare) as of 10/10/2017

Methodology:

*The dataset was split into training set validation set representing respectively 90% and 10% of the dataset

*The input data was standardized (unit standard deviation and null mean)

*All anomalies were removed after plotting the 7 risk factors for all samples

*The missing values in each feature were replaced by the median value of this latter

Regression:

In the first part, we try to consider the problem as a regression problem,and try to predict the target variable through a set of regressors having the following Properties:

Parametric or non-parametric , (L1, L2) penalty term on the cost function or both , Ensemble Methods (Decision Tree Regressor and Extreme Gradient Boosting regressor).

*10fold Cross validation on the regression estimators mentioned above, scored on Explained variance score metric(using make scorer)

Results:

Seems that regression using those risk features wasn’t fruitful, the best score was around 17% Which doesn’t explain the target variable at all.

We see that the Ridge Regressor (L2 regularization) had a better explained variance than the Lasso(L1) because of the nature of the regularization term on Lasso, this latter which takes its abbreviation from Least Absolute Shrinkage and Selector operator, completely annihilate irrelevant features, though we can conclude that the risk factors had about equal contribution to the target variable.

Classification:

Since the regression part wasn't beneficial,we treat the problem as a binary classification,and we try to predict the direction of the target variable.

Methodology:

*10 fold(Stratified) Cross validation on the data using linear/non linear classification models scored on AUC-ROC metric

*Hyperparameter Tuning through GridSearchCV on XGBoost Classifier(sklearn) , and LSVC(sklearn)

*Bagging classifier using (XGBClassifier and LSVC best estimators through GridSearch) , the n_estimators was set to 20,to reduce overfitting,scored on F1 score.

Bagging Results: *F1 score was almost equal to both methods: 0.86

*Mean Accuracy was slighlty better in LSVC(0.77) than XGBclassifier(0.74)

*Using LSVC as an estimator ,has held better overall precision(0.93),but didn't take in consideration the classes respective balances (0.98-Buy / 0.16-Sell) , (Support 283,17),and the probabilities assigned don't have a tangible value,since the Support vector machines are not probabilistic models,so don't provide a predict_probas method

*Using XGBClassifer as an estimator,has held a smaller overall precision (0.70) ,but did take the class balances in consideration and held a better prediction for the sell side (0.78-Buy / 0.47-Sell) , (Support 224,76),and does provide a predict_proba method which can be useful in designing a better portfolio allocation.

Next:

To get better results,one would collect more data,more risk factors and explanatory variables,up to date informations,field knowledge, take in consideration market regime,and try to use deep neural networks for their non linear activation functions,which may explain the stochastic nature of the market and reduce the signal/noise ratio.
