# Factory
from typing import Any, Dict, Optional, Type
from .base import (
    ModelBase,
    LinearRegressionModel,
    RandomForestModel,
    GradientBoostingModel,
)


class ModelFactory:
    """Factory class to create and manage different types of regression models.

    This class implements the Factory design pattern, providing a centralized
    way to create model instances based on type strings. It maintains a registry
    of available model types and supports adding new model types dynamically.

    Attributes:
        _models (Dict[str, Type[ModelBase]]): Registry mapping model type strings
            to their corresponding class implementations.

    Examples:
        >>> factory = ModelFactory()
        >>> lr_model = factory.create_model('linear_regression')
        >>> rf_model = factory.create_model('random_forest', n_estimators=200)
        >>> gb_model = factory.create_model('gradient_boosting')
    """

    _models: Dict[str, Type[ModelBase]] = {
        "linear_regression": LinearRegressionModel,
        "random_forest": RandomForestModel,
        "gradient_boosting": GradientBoostingModel,
    }

    @classmethod
    def create_model(cls, model_type: str, **kwargs: Any) -> ModelBase:
        """Create a model instance of the specified type.

        Args:
            model_type: Type of model to create. Available options are:
                - 'linear_regression': Creates a LinearRegressionModel.
                - 'random_forest': Creates a RandomForestModel.
                - 'gradient_boosting': Creates a GradientBoostingModel.
            **kwargs: Additional parameters to pass to the model constructor.

        Returns:
            A ModelBase instance of the specified type, ready for training.

        Raises:
            ValueError: If the specified model_type is not in the registry.

        Examples:
            >>> model = ModelFactory.create_model('random_forest', n_estimators=200)
            >>> type(model)
            <class '__main__.RandomForestModel'>
        """
        model_class: Optional[Type[ModelBase]] = cls._models.get(model_type.lower())
        if model_class is None:
            available_types: list = list(cls._models.keys())
            raise ValueError(
                f"Unknown model type: '{model_type}'. "
                f"Available types: {available_types}"
            )
        return model_class(**kwargs)

    @classmethod
    def get_available_models(cls) -> list:
        """Get a list of all available model types.

        Returns:
            List of model type strings that can be used with create_model().
        """
        return list(cls._models.keys())
