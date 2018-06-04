# Spam Filtering Techniques for Short Message Service

Adrien Besson and Dimitris Perdios

[Signal Processing Laboratory (LTS5)][lts5],
[École Polytechnique Fédérale de Lausanne (EPFL)][epfl],
Switzerland

Final project for the [Adaptation and Learning][ee621] course given by
[Prof. Ali H. Sayed][sayed], 2018.

> We study various short message service spam filtering techniques based on a
Kaggle dataset composed of 5572 messages, whose 4825 are legitimate and
747 are spam.
The Bag-of-Words models followed by term-frequency-inverse-document-frequency
transformation is employed for feature extraction.
Several state-of-the-art classifiers are compared, i.e. logistic regression,
regularized logistic regression, linear and kernel support vector machine (SVM),
k-nearest neighbours, multinomial Bayes, decision trees, random forests,
AdaBoost and neural networks, where the best hyper-parameters are identified
using 10-fold cross validation.
We demonstrate that all the classifiers perform remarkably well in terms of
misclassification error and that even simple linear methods, such as
logistic regression leads to less than 2% of misclassification error.
We study two reseampling methods that can be used to counter the
class imbalance present in the training set, i.e. downsampling of the
majority class and upsampling of the minority class.
We show that both lead to an increase of the sensitivity at the cost of
a lower specificity.
Online learning strategies are finally investigated, where the algorithms
sequentially update with a new batch of messages, mimicking a more
realistic example.
The supporting code is available at
https://github.com/dperdios/sms-spam-filtering.

[ee621]: https://edu.epfl.ch/coursebook/en/adaptation-and-learning-EE-621
[epfl]: https://www.epfl.ch/
[lts5]: https://lts5www.epfl.ch
[sayed]: https://people.epfl.ch/cgi-bin/people?id=283344&lang=en&cvlang=en

## Installation
1. Install [Python] 3.6 and optionally create a dedicated environment
1. Clone the repository
    ```bash
    git clone https://github.com/dperdios/sms-spam-filtering
    cd sms-spam-filtering
    ```
1. Install the [Python] dependencies from `requirements.txt`
    ```bash
    pip install --upgrade -r requirements.txt 
    ```
[python]: https://www.python.org

## Dataset
We used the [SMS Spam Collection dataset][sms-dataset] distributed
by [kaggle].
> The SMS Spam Collection is a set of SMS tagged messages that have been
collected for SMS Spam research.
It contains one set of SMS messages in English of 5574 messages,
tagged acording being ham (legitimate) or spam.

For simplicity, it is also stored on this repository under `datasets/spam.csv`. 

More info on the dataset: [link][sms-dataset]

[kaggle]: https://www.kaggle.com/
[sms-dataset]: https://www.kaggle.com/uciml/sms-spam-collection-dataset

## Code
The following [Python] scripts and [Jupyter] notebooks are available:

* [data_exploration.ipynb]: Data exploration notebook.
* [example_classifier.ipynb]: Notebook providing an example of classifier
training.
* [increasing_sensitivity.ipynb]: Notebook providing an example of sensitivity
increase by dataset resampling.
* [data_exploration.py]: Produces the data exploration figures (stored under
`results/data-exploration`).
* [classifiers_grid_search.py]: Allows to re-train the classifiers for the
different dataset sampling strategies.
* [classifiers_scores.py]: Allows to compute the different scores on trained
classifiers (stored under `results/trained-classifiers`).
* [online_learning.py]: Allows to re-run the online learning experiments.
Note that the training will only be performed if the configuration is not
already stored under `results/online-learning`

[jupyter]: https://jupyter.org/
[data_exploration.ipynb]: https://nbviewer.jupyter.org/github/dperdios/sms-spam-filtering/blob/scripts/data_exploration.ipynb
[example_classifier.ipynb]: https://nbviewer.jupyter.org/github/dperdios/sms-spam-filtering/blob/scripts/example_classifier.ipynb
[increasing_sensitivity.ipynb]: https://nbviewer.jupyter.org/github/dperdios/sms-spam-filtering/blob/scripts/increasing_sensitivity.ipynb
[data_exploration.py]: scripts/data_exploration.py
[classifiers_grid_search.py]: scripts/classifiers_grid_search.py
[classifiers_scores.py]: scripts/classifiers_scores.py
[online_learning.py]: scripts/online_learning.py

## License
The code is released under the terms of the [MIT license](LICENSE.txt).
