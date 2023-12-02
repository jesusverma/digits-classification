import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import datasets, metrics
from utils import preprocess_data

def read_digits_data():
    """
    Read the digits dataset from scikit-learn.
    """
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target
    return X, y

def save_trained_model(model, filename):
    """
    Save a trained model to a file using joblib.
    """
    joblib.dump(model, filename)

def train_and_evaluate_logistic_regression(X_train, y_train, X_test, y_test, solver):
    """
    Train a logistic regression model, predict, and evaluate its performance.
    """
    # Initialize and train logistic regression model
    clf = LogisticRegression(solver=solver, max_iter=1000)
    clf.fit(X_train, y_train)

    # Predict and evaluate
    predicted = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, predicted)
    print(f"\tLogisticRegression Accuracy with solver '{solver}': {accuracy:.4f}")

    # Cross-validation
    cv_scores = cross_val_score(clf, X, y, cv=5)
    print(f"\tLogisticRegression Mean and Std with 5 crosss CV '{solver}': {cv_scores.mean():.4f}, {cv_scores.std():.4f}")

    # Save the model
    filename = f"m22aie203_logistic_regression_{solver}.joblib"
    save_trained_model(clf, os.path.join("models", filename))
    print(f"\tLogisticRegression Model saved as '{filename}'")

# Read and preprocess the dataset
X, y = read_digits_data()
X = preprocess_data(X)

# Split data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

for solver in solvers:
    print(f"\tLogisticRegression Solver training and eval: {solver}")
    train_and_evaluate_logistic_regression(X_train, y_train, X_test, y_test, solver)
