#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    get_text_vectorizer, load_dataset, logger,
    get_default_classifiers_grid_search, savefig_kwargs
)

###############################################################################
# Data loading
###############################################################################
# Load dataset as DataFrame
filepath = os.path.join(os.pardir, 'datasets', 'spam.csv')
dataset_sample_type = 'normal'
dataset_random_state = 123456
test_size = 0.2

###############################################################################
# Online learning
###############################################################################
# Save options
save_folder = os.path.join(os.pardir, 'results', 'online-learning')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Global cross-validation parameters
random_state = 10
gs_steps = 10
n_jobs = -1
cv = 10  # 10-fold cross-validation

# Online learning parameters
draw_number = 100
update_step = 200
global_save_name = 'online_learning_md{}_us{}'.format(draw_number, update_step)

# Figure paramters
flag_save_fig = False
xlabel = 'Train set size [─]'
ylabel = 'Misclassification error [%]'

# Logistic regression
classifiers_names = ['LR-l2', 'MNB']
classifiers_gscv = get_default_classifiers_grid_search(
    types=classifiers_names, cv=cv, gs_steps=gs_steps, n_jobs=n_jobs,
    random_state=random_state
)

# Initialization
score_array = []
prng = np.random.RandomState(seed=dataset_random_state)
rnd_list = prng.randint(0, 10000, draw_number)
#   Get length list
full_train, full_test = load_dataset(
    filepath, test_size=test_size,
    dataset_sample_type=dataset_sample_type, random_state=None
)
length_list = np.arange(update_step, len(full_train), update_step)
score_dict = dict.fromkeys(classifiers_names)

for clf_nm, clf_gscv in classifiers_gscv.items():
    save_filename = '{}_{}.npy'.format(clf_nm, global_save_name)
    save_file_path = os.path.join(save_folder, save_filename)

    # Multiple draws
    if not os.path.exists(save_file_path):
        logger.info(79 * '/')
        logger.info('Training {}'.format(clf_nm))
        score_list_concat = []

        for rnd in rnd_list:
            logger.info(49 * '#')
            logger.info('Random state: {}'.format(rnd))
            # Load dataset
            full_train, full_test = load_dataset(
                filepath, test_size=test_size,
                dataset_sample_type=dataset_sample_type, random_state=rnd
            )
            # Get test labels
            test_labels = full_test['class']

            score_list = []
            for _len in length_list:
                logger.info('Train set size: {}'.format(_len))

                # Extract sub-train set
                train = full_train[:_len]

                # Feature extraction
                #   Get default text vectorizer
                vectorizer = get_text_vectorizer()
                #   Extract features
                train_features = vectorizer.fit_transform(train['message'])
                test_features = vectorizer.transform(full_test['message'])
                #   Get train labels
                train_labels = train['class']

                clf_gscv.fit(X=train_features, y=train_labels)
                score = clf_gscv.score(X=test_features, y=test_labels)
                score_list.append(score)
                logger.info(
                    '\tClassification loss: {} %'.format((1 - score) * 100)
                )

            # Store result
            score_list_concat.append(score_list)

        # Save
        score_array = np.array(score_list_concat)
        np.save(save_file_path, score_array)
    else:
        score_array = np.load(save_file_path)

    # Store score arrays
    score_dict[clf_nm] = score_array

    # Intermediate plots
    me_array = (1 - score_array) * 100
    me_mean = np.mean(me_array, axis=0)
    me_std = np.std(me_array, axis=0)

    fig_mean = plt.figure()
    ax = fig_mean.add_subplot(111)
    ax.plot(length_list, me_mean, '-o')
    ax.grid()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig_std = plt.figure()
    ax = fig_std.add_subplot(111)
    # ax.plot(length_list, me_mean, '-o')
    (_, caps, _) = ax.errorbar(length_list, me_mean, yerr=2 * me_std, fmt='-o',
                               capsize=3)
    for cap in caps:
        cap.set_markeredgewidth(1)
    ax.grid()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    figname_base = os.path.splitext(save_filename)[0]
    if flag_save_fig:
        fig_mean.savefig(
            os.path.join(save_folder, figname_base) + '_mean.pdf',
            **savefig_kwargs
        )
        fig_std.savefig(
            os.path.join(save_folder, figname_base) + '_mean_std.pdf',
            **savefig_kwargs
        )
    else:
        plt.show()

###############################################################################
# Global plots
###############################################################################
fig_mean = plt.figure()
fig_std = plt.figure()

ax_mean = fig_mean.add_subplot(111)
ax_std = fig_std.add_subplot(111)

for clf_nm, sc_arr in score_dict.items():
    me_array = (1 - sc_arr) * 100
    me_mean = np.mean(me_array, axis=0)
    me_std = np.std(me_array, axis=0)

    ax_mean.plot(length_list, me_mean, '-o', label=clf_nm)
    (_, caps, _) = ax_std.errorbar(
        length_list, me_mean, yerr=2 * me_std, fmt='-o', capsize=3,
        label=clf_nm
    )
    for cap in caps:
        cap.set_markeredgewidth(1)

for ax in [ax_mean, ax_std]:
    ax.legend()
    ax.grid()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

if flag_save_fig:
    fig_mean.savefig(
        os.path.join(save_folder, global_save_name) + '_mean.pdf',
        **savefig_kwargs
    )
    fig_std.savefig(
        os.path.join(save_folder, global_save_name) + '_mean_std.pdf',
        **savefig_kwargs
    )
else:
    plt.show()

logger.info('Done')