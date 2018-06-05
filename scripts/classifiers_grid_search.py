#!/usr/bin/env python3
import os
import pickle
from utils import (
    get_text_vectorizer, load_dataset, logger,
    get_default_classifiers_grid_search
)
from sklearn.metrics import classification_report

###############################################################################
# Data loading
###############################################################################
# Load dataset as DataFrame
filepath = os.path.join(os.pardir, 'datasets', 'spam.csv')
dataset_sample_type = 'normal'
# dataset_sample_type = 'upsample'
# dataset_sample_type = 'downsample'
dataset_random_state = 123456
test_size = 0.2

full_train, full_test = load_dataset(
    filepath, test_size=test_size, dataset_sample_type=dataset_sample_type,
    random_state=dataset_random_state
)

###############################################################################
# Feature extraction
###############################################################################
# Get default text vectorizer
vectorizer = get_text_vectorizer()

# Extract features
#   Fit and transform train set (learns vocabulary + transform)
train_features = vectorizer.fit_transform(full_train['message'])
#   Transform the test set
test_features = vectorizer.transform(full_test['message'])

###############################################################################
# Classifiers grid search definitions
###############################################################################
# Global cross-validation parameters
random_state = 10
gs_steps = 10
n_jobs = -1
cv = 10  # 10-fold cross-validation

classifier_names = [
    'LR', 'LR-l2', 'LR-l1', 'SVM-L', 'SVM-R', 'SVM-S', 'MNB', 'kNN',
    'DT', 'RF', 'AB', 'MLP'
]

classifiers_gscv = get_default_classifiers_grid_search(
    types=classifier_names, cv=cv, gs_steps=gs_steps, n_jobs=n_jobs,
    random_state=random_state
)

###############################################################################
# Classifiers training
###############################################################################
# Save path
save_path = os.path.join(
    os.pardir, 'results', 'trained-classifiers',
    dataset_sample_type + '-dataset'
)

# Check if already exists
if os.path.exists(save_path):
    usr_in = input('Directory `{path}` already exists, '
                   'do you want to continue (y/[n])? '
                   .format(path=save_path))
    usr_in = usr_in.lower() if usr_in else 'n'  # default: 'n'
    if usr_in.lower() != 'y':
        raise InterruptedError
else:
    os.makedirs(save_path)

# Instantiate best dict
best_classifiers = dict.fromkeys(classifiers_gscv.keys())

# Get labels
train_labels = full_train['class']
test_labels = full_test['class']

# Classification report options
digits = 4
target_names = ['ham', 'spam']

for clf_nm, clf_gscv in classifiers_gscv.items():
    logger.info(79 * '#')
    logger.info('Training {}'.format(clf_nm))
    clf_gscv.fit(X=train_features, y=train_labels)
    # Get (and store) best estimator
    best_classifiers[clf_nm] = clf_gscv.best_estimator_
    # Save best estimator
    logger.info('\tSave best estimator:')
    save_nm = '{}_best_estimator.pickle'.format(clf_nm)
    with open(os.path.join(save_path, save_nm), 'wb') as f:
        pickle.dump(clf_gscv.best_estimator_, f, pickle.HIGHEST_PROTOCOL)
    # Test best estimator
    score = clf_gscv.score(X=test_features, y=test_labels)
    logger.info('\tClassification loss: {} %'.format((1 - score) * 100))
    clf_report = classification_report(
        y_true=test_labels, y_pred=clf_gscv.predict(test_features),
        target_names=target_names, digits=digits
    )
    logger.info('\tClassification report:')
    logger.info(clf_report)

logger.info('Done')
