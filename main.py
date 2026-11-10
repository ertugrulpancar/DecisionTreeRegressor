import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.model_selection import GridSearchCV


np.set_printoptions(edgeitems=10, linewidth=200)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

df = pd.read_csv(
    "/Users/ertugrul/PycharmProjects/DecisionTreeRegressor/insurance.csv")
print(df.head())
df.info()
df.describe()

df.shape

df.isnull().sum()

# sex, smoker, region

encode_col = ['sex', 'smoker', 'region']

for col in encode_col:
    print(df[col].value_counts())

for col in encode_col:
    print(df[col].unique())


X = df.drop('charges', axis=1)
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

sex_map = {"female": 0, "male": 1}
smoker_map = {"no": 0, "yes": 1}

X_train = X_train.copy()
X_test = X_test.copy()

for data in [X_train, X_test]:
    data["sex"] = data["sex"].map(sex_map)
    data["smoker"] = data["smoker"].map(smoker_map)

one_hot_encoder = ColumnTransformer(
    transformers=[
        ("region", OneHotEncoder(handle_unknown='ignore'), ['region'])
    ],
    remainder='passthrough'
)

X_train_encoded = one_hot_encoder.fit_transform(X_train)
X_test_encoded = one_hot_encoder.transform(X_test)

one_hot_encoder.get_feature_names_out()
X_train_encoded

tree_model = DecisionTreeRegressor(random_state= 42)
tree_model.fit(X_train_encoded, y_train)
y_pred = tree_model.predict(X_test_encoded)

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

param = {
    "criterion": ["squared_error", "absolute_error"],
    "splitter": ["best", "random"],
    "max_depth": [1, 2, 3, 4, 5, 10, 15],
    "min_samples_split": [2, 3, 5, 7, 10],
}

grid = GridSearchCV(estimator= DecisionTreeRegressor(random_state= 42), param_grid= param, scoring= "r2", n_jobs= -1)
grid.fit(X_train_encoded, y_train)

y_pred_grid = grid.predict(X_test_encoded)

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_grid))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_grid))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred_grid)))
print("R2 Score:", r2_score(y_test, y_pred_grid))

import matplotlib.pyplot as plt

grid.best_params_

plt.figure(figsize=(12,8))

from sklearn import tree

tree.plot_tree(tree_model.fit(X_train_encoded, y_train))