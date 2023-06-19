import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the iris dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42)

# Log the model parameters and accuracy with Autolog
mlflow.sklearn.autolog()

# Train a logistic regression model
with mlflow.start_run() as run:
    lr = LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=1000)
    lr.fit(X_train, y_train)

    # Evaluate the model
    y_pred = lr.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)