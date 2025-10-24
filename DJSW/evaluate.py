import numpy as np

import torch
from models.wmlp import WMLP
from torch.utils.data import DataLoader
from utils.wdataloader import USPS06Dataset
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def balanced_accuracy(y_true, y_pred):
    """
    Calculate balanced accuracy for multi-class classification.
    
    Balanced accuracy is the average of recall obtained on each class.
    It's particularly useful for imbalanced datasets.
    
    Args:
        y_true: Ground truth labels (1D array-like)
        y_pred: Predicted labels (1D array-like)
    
    Returns:
        float: Balanced accuracy score between 0 and 1
    
    Raises:
        AssertionError: If inputs are invalid
        ValueError: If computation fails
    """
    try:
        # Convert to numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Input validation
        assert y_true.ndim == 1, "y_true must be 1-dimensional"
        assert y_pred.ndim == 1, "y_pred must be 1-dimensional"
        assert len(y_true) == len(y_pred), "y_true and y_pred must have same length"
        assert len(y_true) > 0, "Input arrays cannot be empty"
        classes = np.unique(y_true)
        assert len(classes) > 0, "No classes found in y_true"
        
        # Calculate recall for each class
        recalls = []
        for cls in classes:
            cls_mask = (y_true == cls)
            n_cls = np.sum(cls_mask)
            
            if n_cls == 0:
                continue
            
            tp = np.sum((y_true == cls) & (y_pred == cls))
            recall = tp / n_cls
            recalls.append(recall)
        
        assert len(recalls) > 0, "No valid recalls computed"
        
        return np.mean(recalls)
        
    except AssertionError as e:
        raise AssertionError(f"Invalid input: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error computing balanced accuracy: {str(e)}")

def evaluate_model(args):
    """Evaluate a users model on their test data."""
    model_path = args.load_checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data
    if args.username == "waly":
        test_dataset = USPS06Dataset(set_type="test")
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        img_dim = test_dataset.get_input_dim()

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    if args.username == "waly":
        model = WMLP(
            input_dim=img_dim,
            output_dim=7
        )
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    # Evaluation
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Metrics
    mse = mean_squared_error(targets, predictions.argmax(axis=1))
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions.argmax(axis=1))
    r2 = r2_score(targets, predictions.argmax(axis=1))
    bacc = balanced_accuracy(targets, predictions.argmax(axis=1))
    
    print(f"Evaluation Results:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Balanced Accuracy: {bacc:.4f}")
    
    def eval_model(args):
        evaluate_model(args)