# CVした実験結果を吐き出す
from collections import defaultdict
from multiprocessing import cpu_count
import numpy as np
from scipy import sparse as sp
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import rcv1
from sklearn.metrics import classification_report

from cleanlab.noise_generation import (
    generate_noise_matrix,
    generate_noise_matrix_from_trace,
    generate_noisy_labels,
)
from cleanlab.classification import LearningWithNoisyLabels

seed = 42
np.random.seed(seed)

# load data and preprocessing
print("loading data..")
data = rcv1.fetch_rcv1()
mask_col = np.array(
    list(map(lambda x: x.endswith("CAT"), data.target_names))
)  # カテゴリーに分けるのが良さそう
target_names = data.target_names[mask_col]
print(
    "target names", target_names
)  # C→corporate industrial, E→economics, G→goverment, M→わからない...Market?
mask_row = (
    data.target[:, mask_col].toarray().sum(axis=1) == 1
)  # マルチクラスが割り当てられているサンプルは削除
y = data.target[mask_row][:, mask_col]
X = data.data[mask_row]
py = y.toarray().sum(axis=0).reshape(-1)  # given labelの数
print("samples", X.shape[0], "category value counts", py)
y = np.array(y.argmax(axis=1)).reshape(-1)  # one-hot to num


# generate noise matrix
noise_matrix = generate_noise_matrix_from_trace(
    4, 3, min_trace_prob=0.6, frac_zero_noise_rates=0.5, py=py, seed=seed,
)
print("p(given=i|true=j) =")
print(noise_matrix)
"""
p(given=i|true=j) =
[[0.68936167 0.         0.         0.        ]
 [0.2387445  0.85410683 0.21184431 0.05112328]
 [0.         0.14589317 0.78815569 0.28050091]
 [0.07189383 0.         0.         0.66837581]]
"""
# define base Classifier
baseclf = LogisticRegression
params = {
    "solver": "liblinear",
    "multi_class": "auto",
    "verbose": 0,
    "random_state": seed,
}

# define experiments


def normal_learning(X_train, y_train, X_test, y_test):
    clf = baseclf(**params)
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)


def ret_trainedCLclass(X_train, y_train, X_test, y_test):
    model = baseclf(**params)
    clf = LearningWithNoisyLabels(clf=model, seed=seed, n_jobs=cpu_count())
    clf.fit(X_train, y_train)
    return clf


def train_without_noisy_labels(X_train, y_train, X_test, y_test, clf=None):
    if clf is None:
        model = baseclf(**params)
        clf = LearningWithNoisyLabels(clf=model, seed=seed, n_jobs=cpu_count())
        clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)


def train_noisy_to_pseudo(X_train, y_train, X_test, y_test, clf=None):
    model = baseclf(**params)
    if clf is None:
        clf = LearningWithNoisyLabels(clf=model, seed=seed, n_jobs=cpu_count())
        clf.fit(X_train, y_train)

    # trainのcorruptedにだけpseudo
    X_with_noise = X_train[clf.noise_mask]
    y_train_pseudo = y_train_corrupted.copy()
    y_train_pseudo[clf.noise_mask] = clf.predict(X_with_noise)
    # きれいにしたtrain dataでtrain
    model.fit(X_train, y_train_pseudo)

    return model.score(X_test, y_test)


def train_test_and_noisy_to_pseudo(X_train, y_train, X_test, y_test, clf=None):
    model = baseclf(**params)
    if clf is None:
        clf = LearningWithNoisyLabels(clf=model, seed=seed, n_jobs=cpu_count())
        clf.fit(X_train, y_train)

    # trainのcorruptedとtestの両方をpseudoにする
    X_with_noise = X_train[clf.noise_mask]
    y_train_pseudo = y_train_corrupted.copy()
    y_train_pseudo[clf.noise_mask] = clf.predict(X_with_noise)
    y_test_psuedo = clf.predict(X_test)
    y_pseudo = np.hstack([y_train_pseudo, y_test_psuedo])
    X_for_pseudo = sp.vstack([X_train, X_test])

    # pseudo込の全データでtrain
    model.fit(X_for_pseudo, y_pseudo)

    return model.score(X_test, y_test)


# cross validation
cv = KFold(4, shuffle=True, random_state=seed)
result = defaultdict(lambda: [])
for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    y_train_corrupted = generate_noisy_labels(y_train, noise_matrix)

    result["ML:clean"].append(normal_learning(
        X_train, y_train, X_test, y_test))
    result["ML:noisy"].append(
        normal_learning(X_train, y_train_corrupted, X_test, y_test)
    )
    clclf_trained = ret_trainedCLclass(
        X_train, y_train_corrupted, X_test, y_test)
    result["CL:wituout noisy labels"].append(
        train_without_noisy_labels(
            X_train, y_train_corrupted, X_test, y_test, clf=clclf_trained)
    )
    result["CL:pseudo for noisy labels"].append(
        train_noisy_to_pseudo(X_train, y_train_corrupted,
                              X_test, y_test, clf=clclf_trained)
    )
    result["CL:pseudo for noisy labels and test set"].append(
        train_test_and_noisy_to_pseudo(
            X_train, y_train_corrupted, X_test, y_test, clf=clclf_trained)
    )
    print()
    print("end of", i, "th oof.")
    print("result→", result)

print()
print(result)
result = pd.DataFrame(result).T
print()
print(result)
