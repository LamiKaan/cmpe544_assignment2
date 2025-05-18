import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from cvxopt import matrix, solvers
from utils import compile_data_for_SVM
import time


class LinearSoftMarginSVMClassifier:
    def __init__(self, X_train, y_train, fold=5):
        # Initialize training features and labels
        self.X_train = X_train
        self.y_train = y_train
        # Save number of folds for cross-validation
        self.fold = fold
        # Initialize candidate penalty terms (C values) as [0.01, 0.1, 1, 10, 100]
        self.candidate_penalty_terms = [10**(-i) for i in range(-2, self.fold-2)]
        self.best_C = None
        # Initialize dictionary to hold classifiers with different penalty terms
        self.classifiers = {}
        self._initialize_classifiers()
        # The variable to hold the best final classifier
        self.final_classifier = None


    def _initialize_classifiers(self):
        for i in range(self.fold):
            self.classifiers[i] = {
                'weights': None,
                'bias': None,
                'slack_variables': None,
                'penalty_term': None,
                'validation_accuracy': None,
                'training_time': None
            }

    def fit(self):
        best_accuracy = 0

        # Split training data into folds for cross-validation
        splits = list(StratifiedKFold(n_splits=self.fold, shuffle=True, random_state=544).split(self.X_train, self.y_train))
       
        # Build a SVM classifier for each penalty term C
        for i in range(self.fold):
            
            # Get the current classifier and the corresponding penalty term
            classifier = self.classifiers[i]
            C = self.candidate_penalty_terms[i]

            # Use one fold for validation, and other folds for training
            train_indices, validation_indices = splits[i]
            
            X_train = self.X_train[train_indices]
            y_train = self.y_train[train_indices]

            X_validation = self.X_train[validation_indices]
            y_validation = self.y_train[validation_indices]

            # Get the number of training samples and feature dimensions from the training data
            N, d = X_train.shape

            
            # The x vector in the QP problem is of size d + 1 + N, where first d elements are the weights w, the next element is the bias b, and the last N elements are the slack variables xi.

            # P (d + 1 + N, d + 1 +N) matrix in QP objective function, where the first d diagonal elements are 1 for w, the next diagonal element is 0 for b, and the last N diagonal elements are 0 for slack variables.
            P = np.zeros((d + 1 + N, d + 1 + N))
            P[:d, :d] = np.identity(d, dtype=int)

            # q (d + 1 + N, 1) vector in QP objective function, where the first d elements are 0 for w, the next element is 0 for b, and the last N elements are C (penalty term) for slack variables.
            q = np.hstack([np.zeros(d + 1), C * np.ones(N)])

            
            # G_1 matrix (N, d + 1 + N) in QP constraints, where each row consists of d elements of -y_i * X_i (label * feature), so the left (N, d) part (first d columns) of G_1 becomes:
            term1 = -y_train[:, np.newaxis] * X_train
            # next element is -y_i for b in each row, so the next column of G_1 becomes:
            term2 = -y_train[:, np.newaxis]
            # last N elements of each row are all 0, except for the slack variable corresponding to the sample, which is -1. So the right (N, N) part (last N columns) of G_1 becomes:
            term3 = -np.eye(N)

            # Final G_1 matrix (N, d + 1 + N) for the margin constraints is:
            G_1 = np.hstack([term1, term2, term3])

            # And h_1 (N, 1) vector on the right side of the margin constraints is all -1:
            h_1 = -np.ones(N)


            # G_2 matrix (N, d + 1 + N) in QP constraints, where each row consists of d+1 elements of 0 for weights and biases, and following N elements are all 0, except for the slack variable corresponding to the sample, which is -1. So G_2 becomes:
            G_2 = np.zeros((N, d + 1 + N))
            G_2[:, d + 1:] = -np.identity(N)

            # And h_2 (N, 1) vector on the right side of the slack variable constraints is all 0:
            h_2 = np.zeros(N)


            # Combine constraints into a single matrix and vector
            G = np.vstack([G_1, G_2])
            h = np.hstack([h_1, h_2])


            # Solve using the convex optimization library
            # Convert numpy arrays to cvxopt matrices
            P = matrix(P)
            q = matrix(q)
            G = matrix(G)
            h = matrix(h)

            start = time.time()
            print(f"Training classifier {i+1} with penalty term C={C}...")
            # Solve the QP problem using cvxopt solver
            solution = solvers.qp(P, q, G, h)
            print(f"Classifier {i+1} training completed.")
            end = time.time()


            # Get the solution vector x from the result of the QP solver
            x = np.array(solution['x']).flatten()

            # Extract weights, bias and slack variables from x
            w = x[:d]
            b = x[d]
            xi = x[d + 1:]

            # All the slack variables should be >= 0, assert that is the case
            assert np.all(xi >= 0), "Negative slack variables detected, which shouldn't have occurred."

            # Predict the labels for the validation set using the obtained parameters
            y_validation_predicted = np.where(X_validation @ w + b >= 0, 1, -1)
            validation_accuracy = accuracy_score(y_validation, y_validation_predicted)

            # Save the classifier info
            classifier['weights'] = w
            classifier['bias'] = b
            classifier['slack_variables'] = xi
            classifier['penalty_term'] = C
            classifier['validation_accuracy'] = validation_accuracy
            classifier['training_time'] = end - start

            # Check if the current classifier is the best one so far
            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                self.best_C = C

        # Train the final classifier on the entire training set using the best penalty term
        C = self.best_C

        # Get the number of training samples and feature dimensions from the training data
        N, d = self.X_train.shape

        
        # The x vector in the QP problem is of size d + 1 + N, where first d elements are the weights w, the next element is the bias b, and the last N elements are the slack variables xi.

        # P (d + 1 + N, d + 1 +N) matrix in QP objective function, where the first d diagonal elements are 1 for w, the next diagonal element is 0 for b, and the last N diagonal elements are 0 for slack variables.
        P = np.zeros((d + 1 + N, d + 1 + N))
        P[:d, :d] = np.identity(d, dtype=int)

        # q (d + 1 + N, 1) vector in QP objective function, where the first d elements are 0 for w, the next element is 0 for b, and the last N elements are C (penalty term) for slack variables.
        q = np.hstack([np.zeros(d + 1), C * np.ones(N)])

        
        # G_1 matrix (N, d + 1 + N) in QP constraints, where each row consists of d elements of -y_i * X_i (label * feature), so the left (N, d) part (first d columns) of G_1 becomes:
        term1 = -self.y_train[:, np.newaxis] * self.X_train
        # next element is -y_i for b in each row, so the next column of G_1 becomes:
        term2 = -self.y_train[:, np.newaxis]
        # last N elements of each row are all 0, except for the slack variable corresponding to the sample, which is -1. So the right (N, N) part (last N columns) of G_1 becomes:
        term3 = -np.eye(N)

        # Final G_1 matrix (N, d + 1 + N) for the margin constraints is:
        G_1 = np.hstack([term1, term2, term3])

        # And h_1 (N, 1) vector on the right side of the margin constraints is all -1:
        h_1 = -np.ones(N)


        # G_2 matrix (N, d + 1 + N) in QP constraints, where each row consists of d+1 elements of 0 for weights and biases, and following N elements are all 0, except for the slack variable corresponding to the sample, which is -1. So G_2 becomes:
        G_2 = np.zeros((N, d + 1 + N))
        G_2[:, d + 1:] = -np.identity(N)

        # And h_2 (N, 1) vector on the right side of the slack variable constraints is all 0:
        h_2 = np.zeros(N)


        # Combine constraints into a single matrix and vector
        G = np.vstack([G_1, G_2])
        h = np.hstack([h_1, h_2])


        # Solve using the convex optimization library
        # Convert numpy arrays to cvxopt matrices
        P = matrix(P)
        q = matrix(q)
        G = matrix(G)
        h = matrix(h)

        start = time.time()
        print(f"\nTraining best final classifier with penalty term C={C} ...")
        # Solve the QP problem using cvxopt solver
        solution = solvers.qp(P, q, G, h)
        print(f"Training completed.")
        end = time.time()


        # Get the solution vector x from the result of the QP solver
        x = np.array(solution['x']).flatten()

        # Extract weights, bias and slack variables from x
        w = x[:d]
        b = x[d]
        xi = x[d + 1:]

        # All the slack variables should be >= 0, assert that is the case
        assert np.all(xi >= 0), "Negative slack variables detected, which shouldn't have occurred."

        # Predict the labels for the validation set using the obtained parameters
        y_train_predicted = np.where(self.X_train @ w + b >= 0, 1, -1)
        validation_accuracy = accuracy_score(self.y_train, y_train_predicted)

        # Save the classifier info
        self.final_classifier = {'weights': w,
                                 'bias': b,
                                 'slack_variables': xi,
                                 'penalty_term': C,
                                 'training_time': end - start}
            
    def predict(self, X_test):

        y_test_predicted = np.where(X_test @ self.final_classifier['weights'] + self.final_classifier['bias'] >= 0, 1, -1)

        return y_test_predicted


if __name__ == "__main__":
    # Paths to the training data
    train_features_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/train_features.npy"))
    train_labels_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/train_labels.npy"))

    # Paths to the test data
    test_features_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/test_features.npy"))
    test_labels_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/test_labels.npy"))

    # Get data for SVM
    X_train, y_train, X_test, y_test = compile_data_for_SVM(train_features_path, train_labels_path, test_features_path, test_labels_path)

    # Create an instance of the LogisticRegressionClassifier
    linear_soft_SVC = LinearSoftMarginSVMClassifier(X_train, y_train)
    # Fit the model on the training data
    linear_soft_SVC.fit()

    # Predict the labels for the test set
    y_predicted = linear_soft_SVC.predict(X_test)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_predicted)

    # Write results to the output file
    output_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "svm_classifier_results.txt"))
    with open(output_file_path, "w") as output_file:

        for i, classifier in linear_soft_SVC.classifiers.items():
            output_file.write(f"Classifier-{i+1} with C={classifier['penalty_term']}:\n")
            minutes = int(classifier['training_time'] // 60)
            seconds = int(classifier['training_time'] % 60)
            output_file.write(f"  Training Time: {minutes} minutes {seconds} seconds\n")
            output_file.write(f"  Validation Accuracy: {classifier['validation_accuracy']:.4f}\n\n")

        output_file.write(f"\nFinal (best) classifier with C={linear_soft_SVC.final_classifier['penalty_term']}\n")
        minutes = int(linear_soft_SVC.final_classifier['training_time'] // 60)
        seconds = int(linear_soft_SVC.final_classifier['training_time'] % 60)
        output_file.write(f"  Training Time: {minutes} minutes {seconds} seconds\n")
        output_file.write(f"  Test Accuracy: {accuracy:.4f}\n")