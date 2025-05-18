import os
import numpy as np
import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.decomposition import PCA
import time

class FeatureExtractor:
    def __init__(self, train_images_path, test_images_path):
        # Load the images and labels of the dataset
        self.train_images = np.load(train_images_path)
        self.test_images = np.load(test_images_path)
        
        # Get the deep neural network model for feature extraction
        self.model, self.image_preprocessor = self._configure_model()

        # Initialize PCA models for dimensionality reduction
        self.pca = None
        self.pca_whitened = None
        
        # Initialize paths and variables for final feature vectors
        self.train_features_path = os.path.join(os.path.dirname(train_images_path), "train_features.npy")
        self.test_features_path = os.path.join(os.path.dirname(test_images_path), "test_features.npy")

        self.whitened_train_features_path = os.path.join(os.path.dirname(train_images_path), "train_features_whitened.npy")
        self.whitened_test_features_path = os.path.join(os.path.dirname(test_images_path), "test_features_whitened.npy")
        
        self.train_features = None
        self.test_features = None

        self.train_features_whitened = None
        self.test_features_whitened = None

    def _configure_model(self):
        # Load pretrained model weights
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        # Load the transform function for converting images to the input format expected by the model
        image_preprocessor = weights.transforms()

        # Load the model with pretrained weights and set it to evaluation mode
        model = efficientnet_b0(weights=weights)
        model.eval()

        return model, image_preprocessor
    
    def extract_features(self):
        # Define the paths to the files of intermediate representations of the images before obtaining the final features
        train_images_preprocessed_path = os.path.join(os.path.dirname(__file__), "train_images_preprocessed.pt")
        test_images_preprocessed_path = os.path.join(os.path.dirname(__file__), "test_images_preprocessed.pt")
        train_model_features_path = os.path.join(os.path.dirname(__file__), "train_model_features.pt")
        test_model_features_path = os.path.join(os.path.dirname(__file__), "test_model_features.pt")
        
        print("Preprocessing images for transforming to model input format...")
        # Preprocess the images as model inputs
        if not(os.path.exists(train_images_preprocessed_path) or os.path.exists(test_images_preprocessed_path)):
            # Convert the images to PyTorch tensors, and duplicate the single grayscale channel to 3 channels (RGB)
            train_images_tensor = torch.from_numpy(self.train_images)
            train_images_RGB = train_images_tensor.unsqueeze(1).repeat(1, 3, 1, 1)
            
            test_images_tensor = torch.from_numpy(self.test_images)
            test_images_RGB = test_images_tensor.unsqueeze(1).repeat(1, 3, 1, 1)

            # Preprocess the images
            train_images_preprocessed = self.image_preprocessor(train_images_RGB)
            test_images_preprocessed = self.image_preprocessor(test_images_RGB)

            # Save the preprocessed images to disk
            torch.save(train_images_preprocessed, train_images_preprocessed_path)
            torch.save(test_images_preprocessed, test_images_preprocessed_path)
        else:
            # If the preprocessed images already exist, load them from disk
            train_images_preprocessed = torch.load(train_images_preprocessed_path)
            test_images_preprocessed = torch.load(test_images_preprocessed_path)
        print("Preprocessing completed.")


        print("Extracting features using the EfficientNet model...")
        # Extract features using the EfficientNet model
        if not(os.path.exists(train_model_features_path) or os.path.exists(test_model_features_path)):
            # Extract train and test features in batches of size 1000 (loading 20000 train images at once can cause memory issues).
            batch_size = 1000
            train_outputs = []
            test_outputs = []

            with torch.no_grad():
                for i in range(0, len(train_images_preprocessed), batch_size):
                    batch = train_images_preprocessed[i:i + batch_size]
                    output = self.model.features(batch)
                    output = self.model.avgpool(output)
                    output = torch.flatten(output, 1)
                    train_outputs.append(output)
                
            train_model_features = torch.cat(train_outputs, dim=0)

            with torch.no_grad():
                for i in range(0, len(test_images_preprocessed), batch_size):
                    batch = test_images_preprocessed[i:i + batch_size]
                    output = self.model.features(batch)
                    output = self.model.avgpool(output)
                    output = torch.flatten(output, 1)
                    test_outputs.append(output)
                
            test_model_features = torch.cat(test_outputs, dim=0)

            # Save the model features to disk
            torch.save(train_model_features, train_model_features_path)
            torch.save(test_model_features, test_model_features_path)
        else:
            # If the model features already exist, load them from disk
            train_model_features = torch.load(train_model_features_path)
            test_model_features = torch.load(test_model_features_path)
        print("Extraction of model features completed.")

        # EfficientNet model's output features are 1280-dimensional vectors, which are already larger than the original images of 28x28=784 dimensions. We apply a 32 component PCA to reduce the dimensionality of these features and make them more manageable for further analysis.
        train_model_features_np = train_model_features.numpy()
        test_model_features_np = test_model_features.numpy()

        print("Reducing dimensionality of model features...")
        print("Applying PCA...")
        # Apply PCA
        self.pca = PCA(n_components=32, random_state=544)
        self.train_features = self.pca.fit_transform(train_model_features_np)
        self.test_features = self.pca.transform(test_model_features_np)
        # Save the PCA features to disk
        np.save(self.train_features_path, self.train_features)
        np.save(self.test_features_path, self.test_features)

        print("Applying PCA whitening...")
        # Apply whitening to the features
        self.pca_whitened = PCA(n_components=32, whiten=True, random_state=544)
        self.train_features_whitened = self.pca_whitened.fit_transform(train_model_features_np)
        self.test_features_whitened = self.pca_whitened.transform(test_model_features_np)
        # Save the whitened features to disk
        np.save(self.whitened_train_features_path, self.train_features_whitened)
        np.save(self.whitened_test_features_path, self.test_features_whitened)
        print("Dimensionality reduction completed and final features saved.")


if __name__ == "__main__":
    start = time.time()
    print("Feature extraction started...")

    # Train images path
    train_images_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/train_images.npy"))

    # Test images path
    test_images_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/test_images.npy"))

    # Create an instance of the FeatureExtractor class
    feature_extractor = FeatureExtractor(train_images_path, test_images_path)
    # Extract features
    feature_extractor.extract_features()

    print("Feature extraction completed.")
    end = time.time()

    # Write feature extraction results to a file
    output_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "feature_extraction_results.txt"))
    with open(output_file_path, "w") as output_file:

        elapsed_time = end - start
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        output_file.write(f"Feature extraction completed in: {minutes} minutes {seconds} seconds.\n\n")

        output_file.write(f"Retained variance ratio of PCA: {feature_extractor.pca.explained_variance_ratio_.sum():.4f}\n")
        output_file.write(f"Retained variance ratio of PCA whitening: {feature_extractor.pca_whitened.explained_variance_ratio_.sum():.4f}\n")
