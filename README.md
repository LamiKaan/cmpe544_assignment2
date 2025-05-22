# To reproduce results

### 1. Clone repo and set environment

Clone the repo:

```bash
git clone https://github.com/LamiKaan/cmpe544_assignment2.git
```

Navigate inside the root directory:

```bash
cd cmpe544_assignment2
```

Create virtual environment:

```bash
python3 -m venv venv
```

Activate the environment:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Put data files in the proper directory

For feature extractor, svm and kmeans to run properly, the train/test images and corresponding labels for the subset of the Quick Draw dataset needs to be saved under "data/quickdraw_subset_np/" directory.

So, the final directory structure should look like:

```
data
  |__ quickdraw_subset_np
                  |__ train_images.npy
                  |__ train_labels.npy
                  |__ test_images.npy
                  |__ test_labels.npy
```

### 3. Run corresponding python files to reproduce results

For feature extraction (Note: Since this includes a deep learning model, it can take up to 30 minutes for all images. Therefore, I also included the extracted feature data in this repo, to reproduce results quickly for other steps.):

```
python feature_extractor/feature_extractor.py
```

For SVM classifier that is implemented from scratch:

```
python svm_classifier/svm_classifier.py
```

For SVM classifiers of Scikit-learn:

```
python svm_classifier/sklearn_svm_classifier.py
```

For K-Means clustering:

```
python kmeans_clustering/kmeans_clustering.py
```
