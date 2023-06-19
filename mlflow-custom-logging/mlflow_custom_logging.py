import mlflow
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load the iris dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42)

mlflow.set_experiment('Experiment-1')

# Start an MLflow run
with mlflow.start_run():
    # Train a logistic regression model
    dt = DecisionTreeClassifier(max_depth=4)
    dt.fit(X_train, y_train)

    # Evaluate the model
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred,average='micro')

    # Log the model parameters and accuracy manually
    mlflow.log_param("max_depth", 4)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("recall", recall)

