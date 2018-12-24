
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.learning_curve import learning_curve

FTRAIN = 'training.csv'
FTEST = 'test.csv'
FLOOKUP = 'IdLookupTable.csv'

BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64
IMAGE_SIZE = 96
NUM_CHANNELS = 1
SEED = 66478  # Set to None for random seed.
NUM_LABELS = 30
NUM_EPOCHS = 1000
VALIDATION_SIZE = 100  # Size of the validation set.
EARLY_STOP_PATIENCE = 100


def load_data(test=False):
    fname = FTEST if test else FTRAIN
    df = pd.read_csv(fname)

    cols = df.columns[:-1]

    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' ') / 255.0)
    df = df.dropna()

    X = np.vstack(df['Image'])
    X = X.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    if not test:
        # y = (df[cols].values -48) / 48.0
        y = df[cols].values / 96.0
        X, y = shuffle(X, y)
        joblib.dump(cols, 'cols.pkl', compress=3)

    else:
        y = None
    return X, y


def plot_sample(x, y, truth=None):
    img = x.reshape(96, 96)
    plt.imshow(img, cmap='gray')
    if y is not None:
        plt.scatter(y[0::2] * 96, y[1::2] * 96)
    if truth is not None:
        plt.scatter(truth[0::2] * 96, truth[1::2] * 96, c='r', marker='x')
    plt.savefig("img.png")


# Small utility function to evaluate a dataset by feeding batches of data to
# {eval_data} and pulling the results from {eval_predictions}.
# Saves memory and enables this to run on smaller GPUs.
def eval_in_batches(data, sess, eval_prediction, eval_data_node):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
        raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = np.ndarray(shape=(size, NUM_LABELS), dtype=np.float32)
    for begin in xrange(0, size, EVAL_BATCH_SIZE):
        end = begin + EVAL_BATCH_SIZE
        if end <= size:
            predictions[begin:end, :] = sess.run(
                eval_prediction,
                feed_dict={eval_data_node: data[begin:end, ...]})
        else:
            batch_predictions = sess.run(
                eval_prediction,
                feed_dict={eval_data_node: data[-EVAL_BATCH_SIZE:, ...]})
            predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions


def plot_learning_curve(loss_train_record, loss_valid_record):
    plt.figure()
    plt.plot(loss_train_record, label='train')
    plt.plot(loss_valid_record, c='r', label='validation')
    plt.ylabel("RMSE")
    plt.legend(loc='upper left', frameon=False)
    plt.savefig("learning_curve.png")


def generate_submission(test_dataset, sess, eval_prediction, eval_data_node):
    test_labels = eval_in_batches(test_dataset, sess, eval_prediction, eval_data_node)
    test_labels *= 96.0
    test_labels = test_labels.clip(0, 96)

    lookup_table = pd.read_csv(FLOOKUP)
    values = []

    cols = joblib.load('cols.pkl')

    for index, row in lookup_table.iterrows():
        values.append((
            row['RowId'],
            test_labels[row.ImageId - 1][np.where(cols == row.FeatureName)[0][0]],
        ))
    submission = pd.DataFrame(values, columns=('RowId', 'Location'))
    submission.to_csv('submission.csv', index=False)


def generate_learning_curve(estimator, title, scoring,  X, y, ylim=None, cv=None,
                            n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.
    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
    An object of that type which is cloned for each validation.
    title : string
    Title for the chart.
    X : array-like, shape (n_samples, n_features)
    Training vector, where n_samples is the number of samples and
    n_features is the number of features.
    y : array-like, shape (n_samples) or (n_samples, n_features), optional
    Target relative to X for classification or regression;
    None for unsupervised learning.
    ylim : tuple, shape (ymin, ymax), optional
    Defines minimum and maximum yvalues plotted.
    cv : integer, cross-validation generator, optional
    If an integer is passed, it is the number of folds (defaults to 3).
    Specific cross-validation objects can be passed, see
    sklearn.cross_validation module for the list of possible objects
    n_jobs : integer, optional
    Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, scoring=scoring, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    plt.savefig("data/learning_curve.png")


def make_submission(test_labels):
    test_labels *= 96.0
    test_labels = test_labels.clip(0, 96)

    lookup_table = pd.read_csv(FLOOKUP)
    values = []

    cols = joblib.load('data/cols.pkl')

    for index, row in lookup_table.iterrows():
        values.append((
            row['RowId'],
            test_labels[row.ImageId - 1][np.where(cols == row.FeatureName)[0][0]],
        ))
    submission = pd.DataFrame(values, columns=('RowId', 'Location'))
    submission.to_csv('data/submission.csv', index=False)


def load_dataframe(test=False):
    fname = FTEST if test else FTRAIN
    df = pd.read_csv(fname)
    cols = df.columns[:-1]
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' ') / 255.0)

    if not test:
        df[cols] = df[cols].apply(lambda y: y / 96.0)
    return df


def extract_test_data(df):
    X = np.vstack(df['Image'].values)
    X = X.astype(np.float32)
    X = X.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    return X

def extract_train_data(df, flip_indices, cols):
    data = df[list(cols) + ['Image']].copy()
    data = data.dropna()

    X = np.vstack(data['Image'].values)
    X = X.astype(np.float32)
    X = X.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

    y = data[data.columns[:-1]].values
    if len(flip_indices) != 0:
        X_flip = X[:, :, ::-1, :]
        X = np.vstack([X, X_flip])
        y_flip = y.copy()
        y_flip[:, ::2] *= -1
        y_flip[:, ::2] += 1
        for a, b in flip_indices:
            y_flip[:, [a, b]] = y_flip[:, [b, a]]

        y = np.vstack([y, y_flip])

    X, y = shuffle(X, y, random_state=42)  # shuffle train data
    y = y.astype(np.float32)

    return X, y


def create_submission(predicted_labels, columns):
    predicted_labels *= 96.0
    predicted_labels = predicted_labels.clip(0, 96)
    df = pd.DataFrame(predicted_labels, columns=columns)
    lookup_table = pd.read_csv(FLOOKUP)
    values = []

    for index, row in lookup_table.iterrows():
        values.append((
            row['RowId'],
            df.ix[row.ImageId - 1][row.FeatureName],
        ))

    submission = pd.DataFrame(values, columns=('RowId', 'Location'))
    submission.to_csv("data/submission.csv", index=False)


SPECIALIST_SETTINGS = [
    dict(
        columns=(
            'left_eye_center_x', 'left_eye_center_y',
            'right_eye_center_x', 'right_eye_center_y',
        ),
        flip_indices=((0, 2), (1, 3)),
        ),

    dict(
        columns=(
            'nose_tip_x', 'nose_tip_y',
        ),
        flip_indices=(),
        ),

    dict(
        columns=(
            'mouth_left_corner_x', 'mouth_left_corner_y',
            'mouth_right_corner_x', 'mouth_right_corner_y',
            'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
        ),
        flip_indices=((0, 2), (1, 3)),
        ),

    dict(
        columns=(
            'mouth_center_bottom_lip_x',
            'mouth_center_bottom_lip_y',
        ),
        flip_indices=(),
        ),

    dict(
        columns=(
            'left_eye_inner_corner_x', 'left_eye_inner_corner_y',
            'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
            'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
            'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
        ),
        flip_indices=((0, 2), (1, 3), (4, 6), (5, 7)),
        ),

    dict(
        columns=(
            'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
            'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
            'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
            'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
        ),
        flip_indices=((0, 2), (1, 3), (4, 6), (5, 7)),
        ),
    ]