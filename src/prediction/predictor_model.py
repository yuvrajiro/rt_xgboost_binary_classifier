import os
import warnings
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


PREDICTOR_FILE_NAME = "predictor.joblib"


class Classifier:
    """A wrapper class for the X  binary classifier.

    This class provides a consistent interface that can be used with other
    classifier models.
    """

    model_name = "xgboost_binary_classifier"

    def __init__(
        self,
        n_estimators: Optional[int] = 100,
        learning_rate: Optional[float] = 0.1,
        max_depth: Optional[int] = 3,
        colsample_bytree: Optional[float] = 0.8,
        min_child_weight: Optional[int] = 1,
        subsample: Optional[float] = 0.8,
        gamma: Optional[float] = 0,
        reg_lambda: Optional[float] = 1,
        **kwargs,
    ):
        """Construct a new XGBoost binary classifier.

        Args:
            n_estimators (int, optional): The number of trees in the forest.
                Defaults to 100.
            learning_rate (float, optional): Boosting learning rate.
                Defaults to 0.1.
            max_depth (int, optional): Maximum tree depth for base learners.
                Defaults to 3.
            colsample_bytree (float, optional): Subsample ratio of columns when
                constructing each tree. Defaults to 0.8.
            min_child_weight (int, optional): Minimum sum of instance weight (hessian)
                needed in a child. Defaults to 1.
            subsample (float, optional): Subsample ratio of the training instance.
                Defaults to 0.8.
            gamma (float, optional): Minimum loss reduction required to make a further
                partition on a leaf node of the tree. Defaults to 0.
            reg_lambda (float, optional): L2 regularization term on weights.
                Defaults to 1.

        """
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.colsample_bytree = float(colsample_bytree)
        self.min_child_weight = int(min_child_weight)
        self.subsample = float(subsample)
        self.gamma = float(gamma)
        self.reg_lambda = float(reg_lambda)
        self.model = self.build_model()
        self._is_trained = False

    def build_model(self) -> XGBClassifier:
        """Build a new XGBoost binary classifier."""

        model = XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            gamma=self.gamma,
            reg_lambda=self.reg_lambda,
            random_state=0,
        )
        return model

    def fit(self, train_inputs: pd.DataFrame, train_targets: pd.Series) -> None:
        """Fit the XGBoost binary classifier to the training data.

        Args:
            train_inputs (pandas.DataFrame): The features of the training data.
            train_targets (pandas.Series): The labels of the training data.
        """
        self.model.fit(train_inputs, train_targets)
        self._is_trained = True

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict class labels for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted class labels.
        """
        return self.model.predict(inputs)

    def predict_proba(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted class probabilities.
        """
        return self.model.predict_proba(inputs)

    def evaluate(self, test_inputs: pd.DataFrame, test_targets: pd.Series) -> float:
        """Evaluate the XGBoost binary classifier and return the accuracy.

        Args:
            test_inputs (pandas.DataFrame): The features of the test data.
            test_targets (pandas.Series): The labels of the test data.
        Returns:
            float: The accuracy of the XGBoost binary classifier.
        """
        if self.model is not None:
            return self.model.score(test_inputs, test_targets)
        raise NotFittedError("Model is not fitted yet.")

    def save(self, model_dir_path: str) -> None:
        """Save the XGBoost binary classifier to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Classifier":
        """Load the XGBoost binary classifier from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Classifier: A new instance of the loaded XGBoost binary classifier.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    def __str__(self):
        return (
            f"Model name: {self.model_name}("
            f"n_estimators: {self.n_estimators}, "
            f"learning_rate: {self.learning_rate}, "
            f"max_depth: {self.max_depth}, "
            f"colsample_bytree: {self.colsample_bytree}, "
            f"min_child_weight: {self.min_child_weight}, "
            f"subsample: {self.subsample}, "
            f"gamma: {self.gamma}, "
            f"reg_lambda: {self.reg_lambda})"
        )


def train_predictor_model(
    train_inputs: pd.DataFrame, train_targets: pd.Series, hyperparameters: dict
) -> Classifier:
    """
    Instantiate and train the predictor model.

    Args:
        train_inputs (pd.DataFrame): The training data inputs.
        train_targets (pd.Series): The training data labels.
        hyperparameters (dict): Hyperparameters for the classifier.

    Returns:
        'Classifier': The classifier model
    """
    classifier = Classifier(**hyperparameters)
    classifier.fit(train_inputs=train_inputs, train_targets=train_targets)
    return classifier


def predict_with_model(
    classifier: Classifier, data: pd.DataFrame, return_probs=False
) -> np.ndarray:
    """
    Predict class probabilities for the given data.

    Args:
        classifier (Classifier): The classifier model.
        data (pd.DataFrame): The input data.
        return_probs (bool): Whether to return class probabilities or labels.
            Defaults to True.

    Returns:
        np.ndarray: The predicted classes or class probabilities.
    """
    if return_probs:
        return classifier.predict_proba(data)
    return classifier.predict(data)


def save_predictor_model(model: Classifier, predictor_dir_path: str) -> None:
    """
    Save the classifier model to disk.

    Args:
        model (Classifier): The classifier model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Classifier:
    """
    Load the classifier model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Classifier: A new instance of the loaded classifier model.
    """
    return Classifier.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Classifier, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the classifier model and return the accuracy.

    Args:
        model (Classifier): The classifier model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The labels of the test data.

    Returns:
        float: The accuracy of the classifier model.
    """
    return model.evaluate(x_test, y_test)
