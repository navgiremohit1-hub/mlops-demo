
import time
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.svm import SVC


# 1. Load the Iris data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

print("Dataset loaded successfully!")
print("Dataset shape:", X.shape)

# Keep the test data separate while tuning.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# 2. Train a baseline Random Forest
baseline = RandomForestClassifier(random_state=42)
baseline.fit(X_train, y_train)
baseline_predictions = baseline.predict(X_test)
baseline_accuracy = accuracy_score(y_test, baseline_predictions)

print("\nBaseline test accuracy:", round(baseline_accuracy, 4))


# 3. Tune Random Forest with GridSearchCV
# 3 x 4 x 2 = 24 configurations; each uses 5-fold cross-validation.
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 3, 5, 7],
    "min_samples_split": [2, 4],
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)

start = time.perf_counter()
grid_search.fit(X_train, y_train)
grid_seconds = time.perf_counter() - start

grid_model = grid_search.best_estimator_
grid_predictions = grid_model.predict(X_test)
grid_accuracy = accuracy_score(y_test, grid_predictions)

print("\nGridSearchCV")
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation accuracy:", round(grid_search.best_score_, 4))
print("Test accuracy:", round(grid_accuracy, 4))
print("Execution time (seconds):", round(grid_seconds, 3))
print(classification_report(y_test, grid_predictions, target_names=iris.target_names))


# 4. Tune Random Forest with RandomizedSearchCV
param_distributions = {
    "n_estimators": [50, 100, 150, 200, 250],
    "max_depth": [None, 2, 3, 4, 5, 6, 8, 10],
    "min_samples_split": [2, 3, 4, 5],
    "min_samples_leaf": [1, 2, 3],
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=15,
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1,
)

start = time.perf_counter()
random_search.fit(X_train, y_train)
random_seconds = time.perf_counter() - start

random_model = random_search.best_estimator_
random_predictions = random_model.predict(X_test)
random_accuracy = accuracy_score(y_test, random_predictions)

print("\nRandomizedSearchCV")
print("Best parameters:", random_search.best_params_)
print("Best cross-validation accuracy:", round(random_search.best_score_, 4))
print("Test accuracy:", round(random_accuracy, 4))
print("Execution time (seconds):", round(random_seconds, 3))
print(classification_report(y_test, random_predictions, target_names=iris.target_names))


# 5. Compare the baseline and tuned Random Forests on the same test set
print("\nFINAL RANDOM FOREST COMPARISON")
print("Baseline test accuracy:", round(baseline_accuracy, 4))
print("GridSearchCV test accuracy:", round(grid_accuracy, 4))
print("RandomizedSearchCV test accuracy:", round(random_accuracy, 4))


# 6. Additional activity: tune an SVM using macro F1
svm_parameters = [
    {"kernel": ["linear"], "C": [0.1, 1, 10]},
    {"kernel": ["rbf"], "C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
]

svm_search = GridSearchCV(
    estimator=SVC(),
    param_grid=svm_parameters,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1,
)

svm_search.fit(X_train, y_train)
svm_predictions = svm_search.best_estimator_.predict(X_test)

print("\nADDITIONAL SVM COMPARISON")
print("Best SVM parameters:", svm_search.best_params_)
print("Best cross-validation macro F1:", round(svm_search.best_score_, 4))
print("SVM test accuracy:", round(accuracy_score(y_test, svm_predictions), 4))
print(classification_report(y_test, svm_predictions, target_names=iris.target_names))


# 7. Additional activity: compare 3-fold and 5-fold cross-validation
print("\nCROSS-VALIDATION FOLD COMPARISON")
for folds in [3, 5]:
    start = time.perf_counter()
    scores = cross_val_score(
        RandomForestClassifier(random_state=42),
        X_train,
        y_train,
        cv=folds,
        scoring="accuracy",
        n_jobs=-1,
    )
    elapsed = time.perf_counter() - start

    print(
        f"{folds}-fold mean accuracy:",
        round(scores.mean(), 4),
        "| time (seconds):",
        round(elapsed, 3),
    )

print("\nAll optimization experiments completed.")
