import functools

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline


sns.set_style("darkgrid")


def cv_score(estimator, X, y, cv=5, scoring=None):
    """Return the mean score of an estimator and classifier objects X and y using cross validation."""

    scaler = StandardScaler()
    pipeline = Pipeline([('transformer', scaler), ('estimator', estimator)])
    cvs = cross_val_score(estimator=pipeline,
                          X=X,
                          y=y,
                          cv=cv,
                          scoring=scoring)
    return cvs.mean()


def cross_val_metrics(metrics, X, y):
    """TODO

    Parameters
    ----------
    metrics : container
    X : features
    y : target

    Returns
    -------
    (metric_names, cv_metrics) : (list, list)
    """

    cv_X_y = functools.partial(cv_score, X=X, y=y)
    cv_metrics = [functools.partial(cv_X_y, scoring=make_scorer(m)) for m in metrics]
    metric_names = [m.__name__ for m in metrics]
    return metric_names, cv_metrics


def score_estimator(estimator, metrics, X, y):
    """Return dict with name of scorer as key and the score as the value for each scorer.

    Parameters
    ----------
    estimator : instance of BaseEstimator
    metrics : container
    X : features
    y : target

    Returns
    -------
    cross_fold_dict : dict
        {name: cv(estimator)}
    """

    metric_names, cv_metrics = cross_val_metrics(metrics, X, y)
    return {name: cv(estimator) for name, cv in zip(metric_names, cv_metrics)}


def grid_search_cv(estimator, param_grid, X, y, cv=5):
    """Return best_params_ after applying a standard scaler"""

    scaler = StandardScaler()
    grid = make_pipeline(scaler, GridSearchCV(estimator, param_grid=param_grid, cv=cv))
    grid.fit(X, y)
    return grid.named_steps['gridsearchcv'].best_params_


def get_hyperparameters(estimator):
    """TODO

    Parameters
    ----------
    estimator : instance of BaseEstimator

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


def plot_residuals(regression_estimator, X_test, y_test, *, alpha=0.2, dpi=1000, save_path=None):
    """Plot residuals with option to save to disk.

    Parameters
    ----------
    # TODO:
    """

    predictions = regression_estimator.predict(X_test)
    residuals = y_test - predictions

    hyper_params = get_hyperparameters(regression_estimator)
    est_name = type(regression_estimator).__name__
    title = f'Residual Plot: {est_name}({hyper_params})'

    plot = sns.scatterplot(x=predictions.ravel(), y=residuals.ravel(), alpha=alpha)
    plt.title(title)
    plt.xlabel('Predictions')
    plt.ylabel('Residuals')

    if save_path is not None:
        plot.get_figure().savefig(save_path, dpi=dpi)


def plot_roc_calc_auc(classifier_estimator, X_train, X_test, y_train, y_test, *, dpi=1000, save_path=None):
    """Plot the ROC and return the AUC. Also save to disk as png with dpi resolution."""

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
