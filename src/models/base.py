from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression


# Abstract Product
class ModelBase(ABC):
    """Abstract base class for all regression models.

    This class defines the interface that all concrete model implementations
    must follow, providing a consistent API for training and prediction.

    Attributes:
        model: The underlying scikit-learn model instance. Can be any regressor
            that implements fit/predict methods.
        is_trained: Flag indicating whether the model has been fitted to data.
    """

    def __init__(self) -> None:
        """Initialize the base model with default values.

        Sets the model to None and is_trained to False, indicating
        that the model needs to be built and trained before use.
        """
        self.model: Optional[Union[BaseEstimator, RegressorMixin]] = None
        self.is_trained: bool = False

    @abstractmethod
    def build(self) -> None:
        """Build the concrete model instance with configured parameters.

        This method must be implemented by all concrete subclasses to
        instantiate the specific scikit-learn model with the appropriate
        hyperparameters. Called automatically by fit() if model hasn't
        been built yet.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        pass

    def fit(
        self, X_train: npt.NDArray[np.float64], y_train: npt.NDArray[np.float64]
    ) -> "ModelBase":
        """Train the model on the provided data.

        Builds the model if it hasn't been built yet, then fits it
        to the training data. Sets is_trained flag to True after
        successful training.

        Args:
            X_train: Training features array of shape (n_samples, n_features).
            y_train: Training target values array of shape (n_samples,).

        Returns:
            Self, allowing for method chaining after fitting.

        Raises:
            ValueError: If X_train or y_train are empty or have incompatible shapes.
        """
        if self.model is None:
            self.build()
        self.model.fit(X_train, y_train)  # type: ignore[union-attr]
        self.is_trained = True
        return self

    def predict(self, X_test: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Make predictions using the trained model.

        Uses the underlying scikit-learn model to generate predictions
        for the provided test features.

        Args:
            X_test: Test features array of shape (n_samples, n_features).

        Returns:
            Predicted target values array of shape (n_samples,).

        Raises:
            ValueError: If the model hasn't been trained yet. Call fit() first.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X_test)  # type: ignore[union-attr]

    def get_model(self) -> Union[BaseEstimator, RegressorMixin]:
        """Return the underlying scikit-learn model instance.

        Provides access to the raw scikit-learn model for advanced
        operations like feature importance analysis or model inspection.

        Returns:
            The underlying scikit-learn model object (trained or untrained).

        Raises:
            ValueError: If the model hasn't been built yet. Call build() or fit() first.
        """
        if self.model is None:
            raise ValueError(
                "Model has not been built yet. Call build() or fit() first."
            )
        return self.model


# Concrete Products
class LinearRegressionModel(ModelBase):
    """Linear Regression model wrapper.

    Wraps scikit-learn's LinearRegression with no hyperparameters,
    providing a simple baseline regression model.

    Attributes:
        model: The scikit-learn LinearRegression instance.
        is_trained: Flag indicating whether the model has been fitted to data.

    Examples:
        >>> model = LinearRegressionModel()
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(self) -> None:
        """Initialize the Linear Regression model wrapper.

        Creates an empty wrapper that will build the actual LinearRegression
        model when build() or fit() is called.
        """
        super().__init__()

    def build(self) -> None:
        """Build the LinearRegression model instance.

        Creates a new LinearRegression model with default scikit-learn
        parameters (fit_intercept=True, etc.).
        """
        self.model = LinearRegression()


class RandomForestModel(ModelBase):
    """Random Forest Regressor model wrapper.

    Wraps scikit-learn's RandomForestRegressor with configurable
    hyperparameters for ensemble-based regression using random forests.

    Attributes:
        params: Dictionary storing the RandomForestRegressor configuration.
        model: The scikit-learn RandomForestRegressor instance.
        is_trained: Flag indicating whether the model has been fitted to data.

    Examples:
        >>> model = RandomForestModel(n_estimators=200, max_depth=20)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 15,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: Union[str, float, int] = "sqrt",
        random_state: Optional[int] = 42,
        n_jobs: int = -1,
    ) -> None:
        """Initialize Random Forest model with hyperparameters.

        Args:
            n_estimators: The number of trees in the forest. More trees
                generally improve performance but increase computation time.
                Defaults to 100.
            max_depth: The maximum depth of each tree. If None, nodes are
                expanded until all leaves are pure or contain less than
                min_samples_split samples. Defaults to 15.
            min_samples_split: The minimum number of samples required to
                split an internal node. Higher values prevent overfitting.
                Defaults to 5.
            min_samples_leaf: The minimum number of samples required to be
                at a leaf node. Higher values smooth the model. Defaults to 2.
            max_features: The number of features to consider when looking
                for the best split. Can be int, float, "sqrt", or "log2".
                Defaults to "sqrt".
            random_state: Controls the randomness of the estimator for
                reproducibility. Defaults to 42.
            n_jobs: The number of jobs to run in parallel. -1 means using
                all available processors. Defaults to -1.
        """
        super().__init__()
        self.params: Dict[str, Any] = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "random_state": random_state,
            "n_jobs": n_jobs,
        }

    def build(self) -> None:
        """Build the RandomForestRegressor with configured parameters.

        Creates a new RandomForestRegressor instance using the parameters
        stored in self.params dictionary. The model is not trained at this
        point - call fit() to train.
        """
        self.model = RandomForestRegressor(**self.params)


class GradientBoostingModel(ModelBase):
    """Gradient Boosting Regressor model wrapper.

    Wraps scikit-learn's GradientBoostingRegressor with configurable
    hyperparameters for gradient boosting-based regression.

    Attributes:
        params: Dictionary storing the GradientBoostingRegressor configuration.
        model: The scikit-learn GradientBoostingRegressor instance.
        is_trained: Flag indicating whether the model has been fitted to data.

    Examples:
        >>> model = GradientBoostingModel(n_estimators=200, learning_rate=0.1)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        min_samples_split: int = 10,
        min_samples_leaf: int = 4,
        subsample: float = 0.8,
        max_features: float = 0.8,
        random_state: Optional[int] = 42,
    ) -> None:
        """Initialize Gradient Boosting model with hyperparameters.

        Args:
            n_estimators: The number of boosting stages to perform. Gradient
                boosting is fairly robust to over-fitting, so a large number
                usually results in better performance. Defaults to 150.
            learning_rate: Learning rate shrinks the contribution of each
                tree. There is a trade-off between learning_rate and
                n_estimators. Defaults to 0.05.
            max_depth: Maximum depth of the individual regression estimators.
                The maximum depth limits the number of nodes in the tree.
                Defaults to 6.
            min_samples_split: The minimum number of samples required to
                split an internal node. Can be used to control over-fitting.
                Defaults to 10.
            min_samples_leaf: The minimum number of samples required to be
                at a leaf node. A split point at any depth will only be
                considered if it leaves at least this many samples.
                Defaults to 4.
            subsample: The fraction of samples to be used for fitting the
                individual base learners. If smaller than 1.0 this results
                in Stochastic Gradient Boosting. Defaults to 0.8.
            max_features: The fraction of features to consider when looking
                for the best split. If smaller than 1.0, this introduces
                randomness into the model. Defaults to 0.8.
            random_state: Controls the randomness of the estimator for
                reproducibility. Defaults to 42.
        """
        super().__init__()
        self.params: Dict[str, Any] = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "subsample": subsample,
            "max_features": max_features,
            "random_state": random_state,
        }

    def build(self) -> None:
        """Build the GradientBoostingRegressor with configured parameters.

        Creates a new GradientBoostingRegressor instance using the parameters
        stored in self.params dictionary. The model is not trained at this
        point - call fit() to train.
        """
        self.model = GradientBoostingRegressor(**self.params)
