import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from utils import compile_data_for_SVM, upscale_and_enhance_image
import time
import cv2

if __name__ == "__main__":
    # Paths to the training data
    train_features_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/train_features.npy"))
    train_labels_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/train_labels.npy"))
    train_images_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/train_images.npy"))

    # Paths to the test data
    test_features_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/test_features.npy"))
    test_labels_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/test_labels.npy"))
    test_images_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/test_images.npy"))

    # Load the training and test features, labels and images
    train_features = np.load(train_features_path)
    test_features = np.load(test_features_path)

    train_labels = np.load(train_labels_path)
    test_labels = np.load(test_labels_path)

    train_images = np.load(train_images_path)
    test_images = np.load(test_images_path)

    # Filter the training data for the classes "rabbit" and "hand"
    rabbit_train_features = train_features[train_labels == 0]
    rabbit_train_labels = train_labels[train_labels == 0]
    rabbit_test_features = test_features[test_labels == 0]
    rabbit_test_labels = test_labels[test_labels == 0]
    rabbit_train_images = train_images[train_labels == 0]
    rabbit_test_images = test_images[test_labels == 0]

    hand_train_features = train_features[train_labels == 2]
    hand_train_labels = train_labels[train_labels == 2]
    hand_test_features = test_features[test_labels == 2]
    hand_test_labels = test_labels[test_labels == 2]
    hand_train_images = train_images[train_labels == 2]
    hand_test_images = test_images[test_labels == 2]

    # Relabel for SVM (1 for rabbit, -1 for hand)
    rabbit_train_labels[:] = 1
    rabbit_test_labels[:] = 1

    hand_train_labels[:] = -1
    hand_test_labels[:] = -1

    # Create combined train/test sets for SVM
    train_features_svm = np.vstack([rabbit_train_features, hand_train_features])
    train_labels_svm = np.concatenate([rabbit_train_labels, hand_train_labels])
    train_images_svm = np.vstack([rabbit_train_images, hand_train_images])

    test_features_svm = np.vstack([rabbit_test_features, hand_test_features])
    test_labels_svm = np.concatenate([rabbit_test_labels, hand_test_labels])
    test_images_svm = np.vstack([rabbit_test_images, hand_test_images])

    # Rename for SVM classifier
    X_train = train_features_svm
    y_train = train_labels_svm
    I_train = train_images_svm

    X_test = test_features_svm
    y_test = test_labels_svm
    I_test = test_images_svm

    # Calculate the mean/average training image for rabbit and hand classes, and save as file
    rabbit_mean_train_image = np.mean(rabbit_train_images, axis=0)
    normalized_rabbit_mean_train_image = cv2.normalize(rabbit_mean_train_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(os.path.dirname(__file__), "images", "rabbit_mean_train_image.png"), normalized_rabbit_mean_train_image)
    enhanced_img = upscale_and_enhance_image(normalized_rabbit_mean_train_image, scale=10, apply_sharpening=True)
    cv2.imwrite(os.path.join(os.path.dirname(__file__), "images", f"rabbit_mean_train_image_enhanced.png"), enhanced_img)

    hand_mean_train_image = np.mean(hand_train_images, axis=0)
    normalized_hand_mean_train_image = cv2.normalize(hand_mean_train_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(os.path.dirname(__file__), "images", "hand_mean_train_image.png"), normalized_hand_mean_train_image)
    enhanced_img = upscale_and_enhance_image(normalized_hand_mean_train_image, scale=10, apply_sharpening=True)
    cv2.imwrite(os.path.join(os.path.dirname(__file__), "images", f"hand_mean_train_image_enhanced.png"), enhanced_img)


    # Create a SVM classifier using the best parameters from the results of "sklearn_svm_classifier.py"
    classifier = SVC(kernel='rbf', C=1, gamma='auto', random_state=544)
    classifier.fit(X_train, y_train)
    
    # Get the indices of support vectors, and corresponding labels
    support_vector_indices = classifier.support_
    support_vector_labels = y_train[support_vector_indices]
    # Split indices by class
    rabbit_support_indices = support_vector_indices[support_vector_labels == 1]
    hand_support_indices = support_vector_indices[support_vector_labels == -1]

    # Get the margin/distance of support vectors to decision boundary
    rabbit_margins = np.abs(classifier.decision_function(X_train[rabbit_support_indices]))
    hand_margins = np.abs(classifier.decision_function(X_train[hand_support_indices]))
    # Find the indices of 3 smallest margin values for each class (corresponding to support vectors that are closest to the decision boundary)
    rabbit_3_smallest_margins = np.argsort(rabbit_margins)[:3]
    hand_3_smallest_margins = np.argsort(hand_margins)[:3]

    # Get the indinces (in the original train set) of the 3 closest support vectors for each class
    rabbit_3_closest_indices = rabbit_support_indices[rabbit_3_smallest_margins]
    hand_3_closest_indices = hand_support_indices[hand_3_smallest_margins]

    # Get the corresponding images and save as files
    for i, index in enumerate(rabbit_3_closest_indices):
        img = I_train[index]
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(os.path.join(os.path.dirname(__file__), "images", f"rabbit_support_{i+1}.png"), img)

        enhanced_img = upscale_and_enhance_image(img, scale=10, apply_sharpening=True)
        cv2.imwrite(os.path.join(os.path.dirname(__file__), "images", f"rabbit_support_{i+1}_enhanced.png"), enhanced_img)

    for i, index in enumerate(hand_3_closest_indices):
        img = I_train[index]
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(os.path.join(os.path.dirname(__file__), "images", f"hand_support_{i+1}.png"), img)

        enhanced_img = upscale_and_enhance_image(img, scale=10, apply_sharpening=True)
        cv2.imwrite(os.path.join(os.path.dirname(__file__), "images", f"hand_support_{i+1}_enhanced.png"), enhanced_img)


    # Now, also find 3 correctly classified images with largest margins (with least ambiguity and maximum confidence as opposed to closest support vectors) from each class, for comparison with support images

    # Compute decision scores for all training data
    decision_values = classifier.decision_function(X_train)

    # Find correctly classified rabbit samples (label = 1 and score > 0)
    rabbit_indices = np.where(y_train == 1)[0]
    rabbit_correct_mask = decision_values[rabbit_indices] > 0
    rabbit_correct_indices = rabbit_indices[rabbit_correct_mask]
    rabbit_correct_values = decision_values[rabbit_correct_indices]

    # Take top 3 furthest (largest margin) rabbit samples
    rabbit_3_largest_values = np.argsort(-rabbit_correct_values)[:3]
    rabbit_3_furthest_indices = rabbit_correct_indices[rabbit_3_largest_values]

    # Repeat for hand class (label = -1 and score < 0)
    hand_indices = np.where(y_train == -1)[0]
    hand_correct_mask = decision_values[hand_indices] < 0
    hand_correct_indices = hand_indices[hand_correct_mask]
    hand_correct_values = np.abs(decision_values[hand_correct_indices])

    # Take top 3 farthest hand samples
    hand_3_largest_values = np.argsort(-hand_correct_values)[:3]
    hand_3_furthest_indices = hand_correct_indices[hand_3_largest_values]

    # Save furthest correctly classified images
    for i, index in enumerate(rabbit_3_furthest_indices):
        img = I_train[index]
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(os.path.join(os.path.dirname(__file__), "images", f"rabbit_furthest_{i+1}.png"), img)

        enhanced_img = upscale_and_enhance_image(img, scale=10, apply_sharpening=True)
        cv2.imwrite(os.path.join(os.path.dirname(__file__), "images", f"rabbit_furthest_{i+1}_enhanced.png"), enhanced_img)

    for i, index in enumerate(hand_3_furthest_indices):
        img = I_train[index]
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(os.path.join(os.path.dirname(__file__), "images", f"hand_furthest_{i+1}.png"), img)

        enhanced_img = upscale_and_enhance_image(img, scale=10, apply_sharpening=True)
        cv2.imwrite(os.path.join(os.path.dirname(__file__), "images", f"hand_furthest_{i+1}_enhanced.png"), enhanced_img)
