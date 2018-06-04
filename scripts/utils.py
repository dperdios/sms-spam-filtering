import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Tuple, Optional, List
import sklearn.utils
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import sklearn.linear_model as lm
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm, tree, ensemble, neighbors
from sklearn.neural_network import MLPClassifier
import logging


def get_default_classifiers_grid_search(
        types: List[str],
        cv: int = 10,
        gs_steps: int = 10,
        n_jobs: int = -1,
        random_state: Optional[int] = None,
) -> dict:
    # Valid classifier types
    valid_types = [
        'LR', 'LR-l2', 'LR-l1', 'SVM-L', 'SVM-R', 'SVM-S', 'MNB', 'kNN', 'DT',
        'RF', 'AB', 'MLP'
    ]
    (lr_key, lrl2_key, lrl1_key, svml_key, svmr_key, svms_key,
     mnb_key, knn_key, dt_key, rf_key, ab_key, mlp_key) = valid_types
    invalid_check = [t not in valid_types for t in types]
    invalid_types = np.array(types)[invalid_check]
    if any(invalid_check):
        raise ValueError(
            'Unsupported classifiers: {}'.format(', '.join(map(str, invalid_types)))
        )

    # GridSearch common arguments
    gs_kwargs = {'cv': cv, 'n_jobs': n_jobs}

    # Instantiate output dict
    clf_gscv = dict.fromkeys(types)

    # Logistic regression classifiers
    lr_solver = 'liblinear'
    lr_param = [{'C': np.logspace(-4, 9, gs_steps)}]
    if lr_key in types:
        lr = lm.LogisticRegression(
            penalty='l2', solver=lr_solver, random_state=random_state
        )
        clf_gscv[lr_key] = GridSearchCV(lr, [{'C': [1e9]}], **gs_kwargs)
    if lrl2_key in types:
        lrl2 = lm.LogisticRegression(
            penalty='l2', solver=lr_solver, random_state=random_state
        )
        clf_gscv[lrl2_key] = GridSearchCV(lrl2, lr_param, **gs_kwargs)
    if lrl1_key in types:
        lrl1 = lm.LogisticRegression(
            penalty='l1', solver=lr_solver, random_state=random_state
        )
        clf_gscv[lrl1_key] = GridSearchCV(lrl1, lr_param, **gs_kwargs)

    # SVM
    svm_param = [{'C': np.logspace(-4, 9, gs_steps)}]
    if svml_key in types:
        svml = svm.SVC(kernel='linear', random_state=random_state)
        clf_gscv[svml_key] = GridSearchCV(svml, svm_param, **gs_kwargs)
    if svmr_key in types:
        svmr = svm.SVC(kernel='rbf', random_state=random_state)
        clf_gscv[svmr_key] = GridSearchCV(svmr, svm_param, **gs_kwargs)
    if svms_key in types:
        svms = svm.SVC(kernel='sigmoid', random_state=random_state)
        clf_gscv[svms_key] = GridSearchCV(svms, svm_param, **gs_kwargs)

    # Multinomial bayes classifier
    if mnb_key in types:
        mnb_param = [{'alpha': np.logspace(1e-10, 1, gs_steps)}]
        mnb = MultinomialNB()
        clf_gscv[mnb_key] = GridSearchCV(mnb, mnb_param, **gs_kwargs)

    # k-NN
    if knn_key in types:
        knn_param = [
            {'n_neighbors': np.linspace(1, 50, gs_steps).astype(np.int)}
        ]
        knn = neighbors.KNeighborsClassifier(algorithm='auto')
        clf_gscv[knn_key] = GridSearchCV(knn, knn_param, **gs_kwargs)

    # Decision tree
    if dt_key in types:
        dt_param = [
            {'min_samples_leaf': np.linspace(1, 500, gs_steps).astype(np.int)}
        ]
        dt = tree.DecisionTreeClassifier(random_state=random_state)
        clf_gscv[dt_key] = GridSearchCV(dt, dt_param, **gs_kwargs)

    # Random forest
    if rf_key in types:
        rf_param = [
            {'n_estimators': np.linspace(100, 1e3, gs_steps).astype(np.int)}
        ]
        rf = ensemble.RandomForestClassifier(random_state=random_state)
        clf_gscv[rf_key] = GridSearchCV(rf, rf_param, **gs_kwargs)

    # AdaBoost
    if ab_key in types:
        ab_param = [
            {'n_estimators': np.linspace(100, 2e3, gs_steps).astype(np.int)},
        ]
        ab = ensemble.AdaBoostClassifier(random_state=random_state)
        clf_gscv[ab_key] = GridSearchCV(ab, ab_param, **gs_kwargs)

    # MLP
    if mlp_key in types:
        mlp_param = [
            {'hidden_layer_sizes': [(5,), (10,), (20,), (30,), (50,)]}
        ]
        mlp = MLPClassifier(
            learning_rate_init=3e-5, max_iter=200, random_state=random_state
        )
        clf_gscv[mlp_key] = GridSearchCV(mlp, mlp_param, **gs_kwargs)

    return clf_gscv


def plot_confusion_matrix(
        confusion_matrix,
        classes,
        ax: Optional[Axes] = None,
        digits: int = 2
):
    # Rename
    cm = confusion_matrix

    if ax is None:
        ax = plt.gca()

    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    fmt = '.{}f'.format(digits)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")


def load_dataset(
        filepath: str,
        test_size: float = 0.2,
        dataset_sample_type: str = 'normal',
        random_state: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Load DataFrame
    df_full = load_spam_dataframe(filepath)

    # Split train / test
    full_train, test_set = train_test_split(
        df_full, test_size=test_size, random_state=random_state
    )

    # Split train set in ham and spam
    full_train_ham = full_train.loc[full_train['label'] == 'ham']
    full_train_spam = full_train.loc[full_train['label'] == 'spam']

    # Case train sampling type
    if dataset_sample_type.lower() == 'normal':
        train_spam = full_train_spam
        train_ham = full_train_ham
    elif dataset_sample_type.lower() == 'upsample':
        # Upsample the minority class (i.e. spam)
        n_samples = full_train_ham.shape[0]
        train_spam = sklearn.utils.resample(
            full_train_spam, replace=True, n_samples=n_samples,
            random_state=random_state
        )
        train_ham = full_train_ham
    elif dataset_sample_type.lower() == 'downsample':
        # Downsample the majority class (i.e. ham)
        train_ham = full_train_ham.sample(
            full_train_spam.shape[0], axis=0, random_state=random_state
        )
        train_spam = full_train_spam
    else:
        raise ValueError('Unknow dataset sampling type')

    # Concatenate train set ham and spam + shuffle
    train_set = pd.concat([train_ham, train_spam]).sample(
        frac=1, random_state=random_state, axis=0
    )

    return train_set, test_set


def load_spam_dataframe(filepath: str) -> pd.DataFrame:
    # Load dataset
    encoding = 'latin-1'
    df = pd.read_csv(filepath, encoding=encoding, usecols=[0, 1])

    # Rename DataFrame columns with more explicit names
    df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)

    # Convert labels into 0 and 1
    df['class'] = df.label.map({'ham': 0, 'spam': 1})

    return df


def get_text_vectorizer(
        use_tf: bool = True,
        use_idf: bool = True,
        analyzer: Optional[str] = 'word',
        stop_words: Optional[str] = 'english',
        ngram_range: Tuple[int, int] = (1, 1)
):
    # Check inputs
    if use_idf and not use_tf:
        raise ValueError('If use_idf is True, use_tf must also be True')

    # Get text vectorizer
    if use_tf:
        vectorizer = TfidfVectorizer(
            analyzer=analyzer, stop_words=stop_words, ngram_range=ngram_range,
            norm='l2', use_idf=use_idf, smooth_idf=True, sublinear_tf=False
        )
    else:
        vectorizer = CountVectorizer(
            analyzer=analyzer, stop_words=stop_words, ngram_range=ngram_range
        )

    return vectorizer


# Logger
#   Create logger and set level to info
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# http://docs.python-guide.org/en/latest/writing/logging/#logging-in-a-library
logger.addHandler(logging.NullHandler())
#   Create console handler (ch) and set level to info
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
#   Create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#   Add formatter to ch
ch.setFormatter(formatter)
#   Add ch to logger
logger.addHandler(ch)

# Figure saving kwargs
savefig_kwargs = {'dpi': 300, 'transparent': True, 'bbox_inches': 'tight'}
