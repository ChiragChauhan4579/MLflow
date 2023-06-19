import mlflow
from sklearn import datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

dataset = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.4)

def evaluate(actual,pred):
    return accuracy_score(actual,pred)

# mlflow.sklearn.autolog()
with mlflow.start_run():
 clf = XGBClassifier()
 clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
mlflow.log_metric("accuracy", evaluate(y_test, y_pred))
mlflow.xgboost.log_model(clf, "xgb-model")