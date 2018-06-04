import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from utils import (
    load_dataset, get_text_vectorizer, plot_confusion_matrix, savefig_kwargs
)
from sklearn.metrics import confusion_matrix

###############################################################################
# Load test set and extract features
###############################################################################
flag_save_fig = False

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

# Get default text vectorizer
vectorizer = get_text_vectorizer()

# Extract features
#   Fit and transform train set (learns vocabulary + transform)
train_features = vectorizer.fit_transform(full_train['message'])
#   Transform the test set
test_features = vectorizer.transform(full_test['message'])

# Get labels
test_labels = full_test['class']

###############################################################################
# Load best classifiers
###############################################################################
# Save path
save_path = os.path.join(
    os.pardir, 'results', 'trained-classifiers',
    dataset_sample_type + '-dataset'
)

# Check trained classifiers
_file_end = '_best_estimator.pickle'
clf_files = [c for c in os.listdir(save_path) if c.endswith(_file_end)]
clf_files.sort()
clf_names = [c.replace('_best_estimator.pickle', '') for c in clf_files]

# Load classifiers
clf_dict = dict.fromkeys(
    ['estimator', 'ME', 'confusion_matrix', 'sensitivity', 'specificity']
)
classifiers = {key: clf_dict.copy() for key in clf_names}

for key, clf_file in zip(classifiers.keys(), clf_files):
    with open(os.path.join(save_path, clf_file), 'rb') as f:
        classifiers[key]['estimator'] = pickle.load(f)

###############################################################################
# Compute metrics
###############################################################################
for key in classifiers.keys():
    _clf = classifiers[key]['estimator']
    # Miss-classification error
    score = _clf.score(X=test_features, y=test_labels)
    me = (1 - score) * 100
    classifiers[key]['ME'] = me
    # Normalized confusion matrix
    test_preds = _clf.predict(test_features)
    cm = confusion_matrix(y_true=test_labels, y_pred=test_preds)
    classifiers[key]['confusion_matrix'] = cm / cm.sum(axis=1)[:, np.newaxis]
    classifiers[key]['specificity'] = cm[0, 0] / np.sum(cm[0])
    classifiers[key]['sensitivity'] = cm[1, 1] / np.sum(cm[1])

###############################################################################
# Plots
###############################################################################
classes = ['ham', 'spam']
digits = 3

fig = plt.figure(figsize=(12.8, 7.2))
grid = ImageGrid(fig, '111', nrows_ncols=(2, 6), share_all=True,
                 axes_pad=0.3, label_mode="L")
for ax, (clf_nm, clf) in zip(grid, classifiers.items()):
    cm = clf['confusion_matrix']
    plot_confusion_matrix(cm, ax=ax, classes=classes, digits=digits)
    me = clf['ME']
    fmt = '.{}f'.format(digits)
    me_str = format(me, fmt)
    _ttl = '{}: {}%'.format(clf_nm, me_str)
    ax.set_title(_ttl)

figname = '{}_dataset_scores'.format(dataset_sample_type)
export_folder = os.path.join(os.pardir, 'results', 'scores')
if not os.path.exists(export_folder):
    os.makedirs(export_folder)

if flag_save_fig:
    fig.savefig(
        os.path.join(export_folder, figname) + '.pdf', **savefig_kwargs
    )
else:
    plt.show()
