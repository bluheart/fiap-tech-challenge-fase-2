from typing import Optional
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .base import ModelBase


class MLPModel(ModelBase):
    """
    Multi-Layer Perceptron model for rating prediction.

    A PyTorch-based MLP that conforms to the ModelBase interface,
    allowing it to be used interchangeably with scikit-learn models.

    Architecture:
    - Input layer matching feature dimension
    - Configurable hidden layers with BatchNorm, ReLU, and Dropout
    - Linear output layer

    Training features:
    - Adam optimizer with weight decay
    - Learning rate scheduling on plateau
    - Early stopping with patience
    - Automatic validation split if not provided

    Attributes:
        model: The underlying PyTorch Sequential model instance.
        is_trained: Flag indicating whether the model has been fitted.
        input_dim: Input feature dimension (set during fit).
        hidden_dims: List of hidden layer dimensions.
        dropout_rate: Dropout rate for regularization.
        batch_size: Batch size for training.
        epochs: Maximum number of training epochs.
        learning_rate: Learning rate for Adam optimizer.
        weight_decay: L2 regularization strength.
        patience: Early stopping patience.
        device: Device to run the model on (cpu or cuda).
        best_model_state: State dict of the best model during training.
    """

    def __init__(
        self,
        hidden_dims: list = [256, 128, 64],
        dropout_rate: float = 0.3,
        batch_size: int = 256,
        epochs: int = 100,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        patience: int = 10,
        device: str = "cpu",
    ) -> None:
        """
        Initialize the MLP model with configurable architecture and training parameters.

        Args:
            hidden_dims: List of integers specifying the number of units in each hidden layer.
                        Default [256, 128, 64] creates three hidden layers.
            dropout_rate: Dropout probability for regularization (between 0 and 1).
            batch_size: Number of samples per gradient update.
            epochs: Maximum number of complete passes through the training data.
            learning_rate: Step size for the Adam optimizer.
            weight_decay: L2 regularization factor for Adam optimizer.
            patience: Number of epochs to wait for improvement before early stopping.
            device: Device to run the model on ('cpu' or 'cuda').
        """
        super().__init__()
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.device = device
        self.input_dim = None
        self.best_model_state = None
        self.model: Optional[nn.Sequential] = None

    def build(self) -> None:
        """
        Build the PyTorch MLP model architecture.

        Creates a sequential model with:
        - Input layer: Linear(input_dim, hidden_dims[0])
        - Hidden layers: Linear, BatchNorm1d, ReLU, Dropout
        - Output layer: Linear(last_hidden_dim, 1)

        This method is called automatically by fit() if the model
        hasn't been built yet. Requires input_dim to be set.

        Raises:
            ValueError: If input_dim is not set before building.
        """
        if self.input_dim is None:
            raise ValueError("input_dim must be set before building the model")

        layers = []
        prev_dim = self.input_dim

        # Build hidden layers
        for hidden_dim in self.hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_rate),
                ]
            )
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        # Create sequential model and move to device
        self.model = nn.Sequential(*layers).to(self.device)  # type: ignore

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size,).
        """
        assert self.model
        return self.model(x).squeeze(-1)

    def fit(
        self,
        X_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.float64],
        X_val: Optional[npt.NDArray[np.float64]] = None,
        y_val: Optional[npt.NDArray[np.float64]] = None,
        validation_split: float = 0.1,
        verbose: bool = True,
    ) -> "MLPModel":
        """
        Train the MLP model on the provided data.

        If validation data is not provided, a portion of the training data
        will be used for validation based on validation_split.

        Implements:
        - Mini-batch gradient descent
        - Early stopping on validation loss
        - Learning rate reduction on plateau
        - Model checkpointing (saves best model)

        Args:
            X_train: Training features array of shape (n_samples, n_features).
            y_train: Training target values array of shape (n_samples,).
            X_val: Optional validation features array. If None, split from training data.
            y_val: Optional validation target values array. If None, split from training data.
            validation_split: Fraction of training data to use for validation
                            if X_val and y_val are not provided.
            verbose: Whether to print training progress every 10 epochs.

        Returns:
            Self, allowing for method chaining.

        Raises:
            ValueError: If training data is empty or has incompatible shapes.
        """
        # Validate inputs
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Training data cannot be empty")
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must have the same number of samples")

        # Set input dimension from training data
        self.input_dim = X_train.shape[1]

        # Build the model if not already built
        if self.model is None:
            self.build()

        # Create validation split if not provided
        if X_val is None or y_val is None:
            from sklearn.model_selection import train_test_split

            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=validation_split, random_state=42
            )

        assert X_val is not None and y_val is not None, (
            "Validation data should not be None"
        )
        # Get dataset sizes
        n_train_samples = len(X_train)
        n_val_samples = len(X_val)

        # Convert pandas objects to numpy arrays if needed
        # This handles the case where train_test_split returns pandas Series/DataFrame
        X_train_np = np.asarray(X_train, dtype=np.float64)
        y_train_np = np.asarray(y_train, dtype=np.float64).ravel()
        X_val_np = np.asarray(X_val, dtype=np.float64)
        y_val_np = np.asarray(y_val, dtype=np.float64).ravel()

        # Convert to PyTorch tensors and create dataloaders
        X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val_np, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_np, dtype=torch.float32)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Initialize training components
        criterion = nn.MSELoss()
        assert self.model is not None
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )

        # Training loop with early stopping
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self._forward(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_X.size(0)

            train_loss /= n_train_samples

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self._forward(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item() * batch_X.size(0)

            val_loss /= n_val_samples

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save the best model state
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{self.epochs}], "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}"
                )

        # Load the best model state
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        self.is_trained = True
        return self

    def predict(self, X_test: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Make predictions using the trained MLP model.

        Args:
            X_test: Test features array of shape (n_samples, n_features).

        Returns:
            Predicted target values array of shape (n_samples,).

        Raises:
            ValueError: If the model hasn't been trained yet. Call fit() first.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        if self.model:
            self.model.eval()

        # Convert to PyTorch tensor and create dataloader
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        test_dataset = TensorDataset(X_tensor)
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        predictions = []
        with torch.no_grad():
            for (batch_X,) in test_loader:
                outputs = self._forward(batch_X)
                predictions.extend(outputs.cpu().numpy())

        return np.array(predictions, dtype=np.float64)
