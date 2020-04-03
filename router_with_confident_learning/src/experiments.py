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


'''
defaultdict(<function <lambda> at 0x7ff88dbb2598>, {
'ML:clean': [0.9738480042973585, 0.9745019501599832, 0.9745194665670178, 0.9748112596121845], 
'ML:noisy': [0.9529334142980592, 0.952980124716818, 0.9543931148842749, 0.9542235223364688], 
'CL:wituout noisy labels': [0.959461195319616, 0.9594787117266507, 0.9602961440549315, 0.9599747762265936], 
'CL:pseudo for noisy labels': [0.9618901370950791, 0.9624214681084616, 0.9628944110983955, 0.9633145906683717], 
'CL:pseudo for noisy labels and test set': [0.9620711399677698, 0.9622521428404606, 0.962789312656188, 0.9631744585938914]})
'''

'''
きれいに書き直すとこんな感じ
| method\accuracy                    | test1      | test2      | test3      | test4      | mean  (std)         |
| ---------------------------------- | ---------- | ---------- | ---------- | ---------- | ------------------- |
| ML:clean  (ideal performance)      | 0.9738     | 0.9745     | 0.9745     | 0.9748     | 0.9744 (0.0004)     |
|                                    |            |            |            |            |                     |
| ML:noisy   (baseline performance)  | 0.9529     | 0.9529     | 0.9543     | 0.9542     | 0.9536 (0.0007)     |
| CL:wituout noisy labels            | 0.9594     | 0.9594     | 0.9602     | 0.9599     | 0.9598 (0.0004)     |
| CL:pseudo for noises               | 0.9618     | **0.9624** | **0.9628** | **0.9633** | **0.9626** (0.0006) |
| CL:pseudo for noises and test      | **0.9620** | 0.9622     | 0.9627     | 0.9631     | **0.9625** (0.0005) |
'''
