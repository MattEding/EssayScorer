import functools

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline


def cross_val_metrics(metrics, X, y):
    """Return tuple pair of metric names and their mean  scores from cross validation.

    Parameters
    ----------
    metrics : container
        Container of metric functions.
    X : array-like or sparse matrix, shape (n_samples, n_features)
        Training data.
    y : array_like, shape (n_samples, n_targets)
        Target values. Will be cast to X’s dtype if necessary.

    Returns
    -------
    (metric_names, cv_metrics) : (list, list)
        Tuple of metric names and their corresponding scores.
    """

    cv_X_y = functools.partial(cv_score, X=X, y=y)
    cv_metrics = [functools.partial(cv_X_y, scoring=make_scorer(m)) for m in metrics]
    metric_names = [m.__name__ for m in metrics]
    return metric_names, cv_metrics


def cv_score(estimator, X, y, cv=5, scoring=None):
    """Return the mean score of an estimator and classifier objects X and y using cross validation.

    Parameters
    ----------
    estimator : instance of BaseEstimator
        An estimator instance.
    X : array-like or sparse matrix, shape (n_samples, n_features)
        Training data.
    y : array_like, shape (n_samples, n_targets)
        Target values. Will be cast to X’s dtype if necessary.
    cv : int, cross-validation generator or an iterable, optional

    scoring : string, callable, list/tuple, dict or None, default: None
        A single string or a callable to evaluate the predictions on the test set.
        For evaluating multiple metrics, either give a list of (unique) strings or a dict with names as keys and callables as values.
        If None, the estimator’s default scorer (if available) is used.
    """

    scaler = StandardScaler()
    pipeline = Pipeline([('transformer', scaler), ('estimator', estimator)])
    cvs = cross_val_score(estimator=pipeline,
                          X=X,
                          y=y,
                          cv=cv,
                          scoring=scoring)
    return cvs.mean()


def get_hyperparameters(estimator):
    """Return string of the hyper-parameters passed to the estimator.

    Parameters
    ----------
    estimator : instance of BaseEstimator
        An estimator instance.

    Returns
    -------
    hyper_params : str
        String of the hyper parameters the estimator was instantiated with.
    """

    est_name = type(estimator).__name__
    default_params = type(estimator)().get_params().items()
    estimator_params = estimator.get_params().items()
    hyper_params = (est_p for def_p, est_p in zip(default_params, estimator_params) if def_p != est_p)
    hyper_params = (f'{arg}={val}' for arg, val in hyper_params)
    hyper_params = ', '.join(hyper_params)
    return hyper_params


def grid_search_cv(estimator, param_grid, X, y, cv=5):
    """Return best_params_ after applying a standard scaler.

    Parameters
    ----------
    estimator : instance of BaseEstimator
        An estimator instance.
    param_grid : dict or list of dictionaries
        Dictionary with parameters names as keys and lists of parameter settings
        to try as values, or a list of such dictionaries, in which case the grids
        spanned by each dictionary in the list are explored. This enables searching
        over any sequence of parameter settings.
    X : array-like or sparse matrix, shape (n_samples, n_features)
        Training data.
    y : array_like, shape (n_samples, n_targets)
        Target values. Will be cast to X’s dtype if necessary.
    cv : int, cross-validation generator or an iterable, optional

    Returns
    -------
    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.
    """

    scaler = StandardScaler()
    grid = make_pipeline(scaler, GridSearchCV(estimator, param_grid=param_grid, cv=cv))
    grid.fit(X, y)
    return grid.named_steps['gridsearchcv'].best_params_


def plot_residuals(fit_estimator, X, y, *, alpha=0.2, dpi=500, save_path=None):
    """Plot residuals with option to save to disk.

    Parameters
    ----------
    fit_estimator : instance of BaseEstimator
        An estimator that has been fit.
    X : array-like or sparse matrix, shape (n_samples, n_features)
        Training data.
    y : array_like, shape (n_samples, n_targets)
        Target values. Will be cast to X’s dtype if necessary.
    alpha : float
        Alpha level for scatterplot
    """

    predictions = fit_estimator.predict(X)
    residuals = y - predictions

    hyper_params = get_hyperparameters(fit_estimator)
    est_name = type(fit_estimator).__name__
    title = f'Residual Plot: {est_name}({hyper_params})'

    plot = sns.scatterplot(x=predictions.ravel(), y=residuals.ravel(), alpha=alpha)
    plt.title(title)
    plt.xlabel('Predictions')
    plt.ylabel('Residuals')

    if save_path is not None:
        plot.get_figure().savefig(save_path, dpi=dpi)


def plot_roc_calc_auc(classifier_estimator, X_train, X_test, y_train, y_test, *, dpi=1000, save_path=None):
    """Plot the ROC and return the AUC. Also save to disk as png with dpi resolution.

    Parameters
    ----------
    classifier_estimator : instance of ClassifierMixin
        A classifier estimator instance.
    X_train : array-like or sparse matrix, shape (n_samples, n_features)
        Training data.
    X_test : array-like or sparse matrix, shape (n_samples, n_features)
        Testing data.
    y_train : array_like, shape (n_samples, n_targets)
        Training target values. Will be cast to X’s dtype if necessary.
    y_test : array_like, shape (n_samples, n_targets)
        Testing target values. Will be cast to X’s dtype if necessary.
    dpi : int, optional
        Dots per inch to save plot if save_path is provided.
    save_path : path, optional
        Destination to save plot. If None, do not save.
    """

    estimator.fit(X_train, y_train)
    y_score = estimator.predict_proba(X_test)
    false_pos, true_pos, _ = roc_curve(y_test, y_score[:,1])

    hyper_params = get_hyperparameters(classifier_estimator)
    est_name = type(estimator).__name__
    title = f'ROC: {est_name}({hyper_params})'

    plt.plot(false_pos, true_pos)
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi)

    return f'AUC: {auc(false_pos, true_pos)}'


def score_estimator(estimator, metrics, X, y):
    """Return dict with name of scorer as key and the score as the value for each scorer.

    Parameters
    ----------
    estimator : instance of BaseEstimator
        An estimator instance.
    metrics : container
        Container of metric functions.
    X : array-like or sparse matrix, shape (n_samples, n_features)
        Training data.
    y : array_like, shape (n_samples, n_targets)
        Target values. Will be cast to X’s dtype if necessary.

    Returns
    -------
    cross_fold_dict : dict
        {name: cv(estimator)}
    """

    metric_names, cv_metrics = cross_val_metrics(metrics, X, y)
    return {name: cv(estimator) for name, cv in zip(metric_names, cv_metrics)}
