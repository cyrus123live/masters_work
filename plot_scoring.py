import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pandas.plotting import autocorrelation_plot
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from pygam import LinearGAM, s
import pickle


scoring = {
    "rounds": [],
    "scores": [],
    "trading_scores": []
}
scoring_testing = {
    "rounds": [],
    "scores": [],
    "trading_scores": []
}
with open("scoring.txt", 'r') as f:
    for line in f.readlines():
        scoring["rounds"].append(int(line.split("/")[0].split(" ")[-1]))
        scoring["scores"].append(float(line.split(",")[0].split(" ")[-1]))
        scoring["trading_scores"].append(float(line.split(" ")[-1]))
scoring = pd.DataFrame(scoring)
with open("scoring_testing.txt", 'r') as f:
    for line in f.readlines():
        scoring_testing["rounds"].append(int(line.split("/")[0].split(" ")[-1]))
        scoring_testing["scores"].append(float(line.split(",")[0].split(" ")[-1]))
        scoring_testing["trading_scores"].append(float(line.split(" ")[-1]))
scoring_testing = pd.DataFrame(scoring_testing)
# print(scoring) 
# print(scoring_testing)
figure = plt.figure()
p = figure.add_subplot()

to_plot = pd.DataFrame(index=scoring["rounds"])
to_plot["scores"] = scoring["scores"]
to_plot["trading_scores"] = scoring["trading_scores"] 
# to_plot["trading_scores"] = to_plot["trading_scores"] - scoring["trading_scores"].iloc[-1]
# to_plot["trading_scores"] = to_plot["trading_scores"] / to_plot["trading_scores"].mean()

p.plot(to_plot['scores'], label="scores")
p.plot(to_plot['trading_scores'], label="trading_scores")

p.legend()
plt.show()

# Features and target
X_train = scoring[["scores", "rounds"]].values
y_train = scoring["trading_scores"].values
X_test = scoring_testing[["scores", "rounds"]].values
y_test = scoring_testing["trading_scores"].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Fit GAM model
gam = LinearGAM(s(0) + s(1), n_splines=20).fit(X_scaled, y_train)

# Predict and evaluate
y_pred = gam.predict(X_scaled)
mse = mean_squared_error(y_train, y_pred)
print(f'Mean Squared Error: {mse:.4f}')

# Residual plot
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, y_pred - y_train, alpha=0.7)
plt.hlines(0, min(y_pred), max(y_pred), colors='r')
plt.xlabel('Predicted Trading Score')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Partial dependence plots
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
titles = ['Score', 'Scoring Round']
for i, ax in enumerate(axs):
    XX = gam.generate_X_grid(term=i)
    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=0.95)[1], c='r', ls='--')
    ax.set_title(f'Partial Dependence of {titles[i]}')
    ax.set_xlabel(titles[i])
    ax.set_ylabel('')
plt.tight_layout()
plt.show()

# prediction = pd.DataFrame()
# new_data_scaled = scaler.transform(X_test)
# prediction["pred"] = gam.predict(new_data_scaled)
# prediction["actual"] = y_test
# print(prediction[prediction["pred"] == prediction["pred"].max()])
# print('Predicted Trading Scores:', predictions)

# Save the GAM model
with open('gam_model.pkl', 'wb') as model_file:
    pickle.dump(gam, model_file)

# Save the scaler
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler saved successfully.")

# scoring = pd.DataFrame(scoring)

# X_train = scoring[["scores", "rounds"]].iloc[:1965]
# y_train = scoring["trading_scores"].iloc[:1965]
# X_test = scoring[["scores", "rounds"]].iloc[1965:]
# y_test = scoring["trading_scores"].iloc[1965:]


# param_grid = {
#     'n_estimators': [1, 10, 25, 50, 100, 200],
#     'max_depth': [1, 3, 10, 20, 30, None],
#     'min_samples_split': [2, 5, 10, 20, 50, 100]
# }

# grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
# grid_search.fit(X_train, y_train)

# print("Best parameters found:", grid_search.best_params_)
# print("Best cross-validated MSE:", -grid_search.best_score_)

# params = {'max_depth': 3, 'min_samples_split': 10, 'n_estimators': 1}

# model = RandomForestRegressor(n_estimators=params["n_estimators"], min_samples_split=params["min_samples_split"], max_depth=params["max_depth"])
# model.fit(X_train, y_train)
# prediction = pd.DataFrame()
# prediction["pred"] = model.predict(X_test)
# prediction["actual"] = y_test.values
# print(prediction[prediction["pred"] > 0])

# poly = PolynomialFeatures(degree=1)
# X_poly = poly.fit_transform(X_train)
# poly_model = LinearRegression()
# poly_model.fit(X_poly, y_train)

# X_pol_test = poly.fit_transform(X_test)
# prediction["pred"] = poly_model.predict(X_pol_test)
# prediction["actual"] = y_test.values
# print(prediction[prediction["pred"] == prediction["pred"].max()])

# X_grid = np.arange(min(X),max(X),0.01)
# X_grid = X_grid.reshape(len(X_grid),1) 
  
# plt.scatter(X_test["scores"],y_test, color='blue') #plotting real points
# plt.plot(X_test["scores"], model.predict(X_test),color='green') #plotting for predict points
  
# plt.title("Random Forest Regression Results")
# plt.xlabel('Score')
# plt.ylabel('Trading')
# plt.show()

# plt.scatter(X, y, color='blue')
# plt.plot(X, model.predict(X), color='red')
# plt.xlabel('Training Score')
# plt.ylabel('Trading Score')
# plt.title('Linear Regression Fit')
# plt.show()

# import seaborn as sns

# residuals = y - model.predict(X)
# sns.residplot(x=model.predict(X), y=residuals, lowess=True)
# plt.xlabel('Predicted Trading Score')
# plt.ylabel('Residuals')
# plt.title('Residual Plot')
# plt.show()

