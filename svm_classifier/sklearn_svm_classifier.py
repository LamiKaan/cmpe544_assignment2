import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from utils import compile_data_for_SVM
import time


class SklearnSVMClassifier:
    def __init__(self, X_train, y_train, kernel_list=['linear', 'rbf', 'poly', 'sigmoid']):
        # Initialize training features and labels
        self.X_train = X_train
        self.y_train = y_train
        # Initialize kernel list
        self.kernel_list = kernel_list
        # Initialize results dictionary to store best parameters, training time, and test accuracy for each kernel
        self.results = {}

    def fit_and_evaluate(self, X_test, y_test):
        # For each kernel in the kernel list, train the SVM classifier and evaluate its performance
        for kernel in self.kernel_list:
            print(f"Training SVM with '{kernel}' kernel...")
            start = time.time()

            # Set the parameter grid for GridSearchCV based on the kernel type
            if kernel == 'linear':
                param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
            elif kernel == 'rbf' or kernel == 'sigmoid':
                param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.01, 0.1, 1]}
            elif kernel == 'poly':
                param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'degree': [2, 3, 4], 'gamma': ['scale', 'auto', 0.01, 0.1, 1]}
            else:
                continue

            # Create a GridSearchCV object to find the best hyperparameters for the SVM classifier
            clf = GridSearchCV(
                SVC(kernel=kernel, random_state=544),
                param_grid,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=544),
                scoring='accuracy',
                verbose=0,
                n_jobs=-1
            )

            # Fit the classifier to the training data
            clf.fit(self.X_train, self.y_train)
            end = time.time()

            # Get the classifier with the best parameters and evaluate its performance on the test set
            best_model = clf.best_estimator_
            y_pred = best_model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)

            # Store the results for the current kernel
            self.results[kernel] = {
                'best_params': clf.best_params_,
                'training_time': end - start,
                'test_accuracy': test_accuracy
            }

            print(f"Completed training with '{kernel}' kernel.")

    def save_results(self, output_path):
        with open(output_path, "w") as f:
            for kernel, result in self.results.items():
                f.write(f"SVM with '{kernel}' kernel\n")
                
                minutes = int(result['training_time'] // 60)
                seconds = int(result['training_time'] % 60)
                
                f.write(f"  Best Parameters: {result['best_params']}\n")
                f.write(f"  Training Time: {minutes} minutes {seconds} seconds\n")
                f.write(f"  Test Accuracy: {result['test_accuracy']:.4f}\n\n")


if __name__ == "__main__":
    # Paths to the training data
    train_features_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/train_features.npy"))
    train_labels_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/train_labels.npy"))

    # Paths to the test data
    test_features_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/test_features.npy"))
    test_labels_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/test_labels.npy"))

    # Get data for SVM
    X_train, y_train, X_test, y_test = compile_data_for_SVM(train_features_path, train_labels_path, test_features_path, test_labels_path)

    # Create an instance of the SklearnSVMClassifier
    sklearn_SVC = SklearnSVMClassifier(X_train, y_train)
    # Fit and evaluate the classifiers
    sklearn_SVC.fit_and_evaluate(X_test, y_test)

    # Save results
    output_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "sklearn_svm_classifier_results.txt"))
    sklearn_SVC.save_results(output_file_path)