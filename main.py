import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import os
import pickle


iris = load_iris()
X = iris.data
target = iris.target
names = iris.target_names

df = pd.DataFrame(X, columns=iris.feature_names)
df["species"] = iris.target
df["species"] = df["species"].replace(
    to_replace=[0, 1, 2], value=["setosa", "versicolor", "virginica"]
)
# print(df.head(3))

dict = {"setosa": 0, "versicolor": 1, "virginica": 2}
df["species"] = df["species"].map(dict)

X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.1, random_state=42
)
dt = DecisionTreeClassifier()
model = dt.fit(X_train, y_train)
print(f"Test Score: {model.score(X_test, y_test)}")
preds = model.predict(X_test)
print(f"prediction: {preds}")

# describe the data
df.describe()
# plot seaborn chart
train, test = train_test_split(df, test_size = 0.4, stratify = df['species'], random_state = 42)
sns.pairplot(train, hue="species", height = 2, palette = 'colorblind');

if not os.path.exists("models"):
    os.makedirs("models")

pickle.dump(model, open("models/model.pkl", "wb"))
