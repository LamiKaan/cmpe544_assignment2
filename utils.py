import os
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment


def compile_data_for_SVM(train_features_path, train_labels_path, test_features_path, test_labels_path):
    """
    Load the training and test features and labels, and filter them for the classes "rabbit" and "hand". For the SVM classifier.
    """
    # Load the training and test features and labels
    train_features = np.load(train_features_path)
    test_features = np.load(test_features_path)

    train_labels = np.load(train_labels_path)
    test_labels = np.load(test_labels_path)

    # Filter the training and test features and labels for the classes "rabbit" and "hand"
    rabbit_train_features = train_features[train_labels == 0]
    rabbit_train_labels = train_labels[train_labels == 0]
    rabbit_test_features = test_features[test_labels == 0]
    rabbit_test_labels = test_labels[test_labels == 0]

    hand_train_features = train_features[train_labels == 2]
    hand_train_labels = train_labels[train_labels == 2]
    hand_test_features = test_features[test_labels == 2]
    hand_test_labels = test_labels[test_labels == 2]

    # Relabel for SVM (1 for rabbit, -1 for hand)
    rabbit_train_labels[:] = 1
    hand_train_labels[:] = -1
    rabbit_test_labels[:] = 1
    hand_test_labels[:] = -1

    # Create combined train/test sets for SVM
    train_features_svm = np.vstack([rabbit_train_features, hand_train_features])
    train_labels_svm = np.concatenate([rabbit_train_labels, hand_train_labels])

    test_features_svm = np.vstack([rabbit_test_features, hand_test_features])
    test_labels_svm = np.concatenate([rabbit_test_labels, hand_test_labels])

    # Rename for SVM classifier
    X_train = train_features_svm
    y_train = train_labels_svm

    X_test = test_features_svm
    y_test = test_labels_svm


    return X_train, y_train, X_test, y_test


def upscale_and_enhance_image(image, scale=10, apply_sharpening=True):
    # Resize using high-quality interpolation
    upscaled = cv2.resize(image, (image.shape[1]*scale, image.shape[0]*scale), interpolation=cv2.INTER_LANCZOS4)

    if apply_sharpening:
        # Apply a sharpening kernel
        kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])
        upscaled = cv2.filter2D(upscaled, -1, kernel)

    return upscaled


def calculate_clustering_accuracy(y_true, y_pred):
    # Build a confusion matric of true vs. predicted labels
    confusion = confusion_matrix(y_true, y_pred)
    # Use the Hungarian algorithm to solve the assignment problem between true and predicted labels (maximize the correct matches)
    row_ind, col_ind = linear_sum_assignment(-confusion)
    # Get the total number of correct matches and normalize by the total number of samples
    return confusion[row_ind, col_ind].sum() / y_true.size