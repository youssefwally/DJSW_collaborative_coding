import numpy as np

import torch
from models.wmlp import WMLP
from torch.utils.data import DataLoader
from utils.wdataloader import USPS06Dataset
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def recall(y_true, y_pred, average="macro"):
    """
    Calculate recall for multi-class classification.
        
    Ratio of relevant retrived instances to number of relevant instances.
    
    Args:
        y_true: Ground truth labels (1D array-like)
        y_pred: Predicted labels (1D array-like)
        average: Averaging mode, macro or micro (str)
        
    Returns:
        float: Recall score between 0 and 1
        
    Raises:
        AssertionError: If inputs are invalid
        ValueError: If computation fails
    """
    try:
        # Ensure numpy array
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Validate input
        assert y_true.ndim == 1, "y_true must be 1-dimensional"
        assert y_pred.ndim == 1, "y_pred must be 1-dimensional"
        assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length"
        assert len(y_true) > 0, "Input arrays cannot be empty"
        
        # Get unique classes
        classes = np.unique(np.concatenate((y_true, y_pred)))
         
        # Calculate true positives (tp) and false negatives (fn) for each class
        tps = []
        fns = []        
        for cl in classes:
            tp = np.sum((y_true == cl) & (y_pred == cl))
            fn = np.sum((y_true == cl) & (y_pred != cl))
            tps.append(tp) 
            fns.append(fn)
        
        assert isinstance(average, str), "average (argument) needs to be a string (str)"
        
        # For ease of computation of mean
        tps = np.asarray(tps)
        fns = np.asarray(fns)
        
        if average.lower() == "macro":
            # Make sure to not divide by zero
            recalls = np.divide(tps, tps + fns, out=np.zeros_like(tps, dtype=float), where=(tps + fns) != 0)
            recall = np.mean(recalls)
        elif average.lower() == "micro":
            relevant = np.sum(tps) + np.sum(fns) 
            if relevant == 0:
                recall = 0
            else:
                recall = np.sum(tps)/(relevant)
        else:
            raise ValueError(f"average should be 'macro' or 'micro', {average} was given")       
        
        assert 0 <= recall <= 1, f"Recall score is invalid: {recall:.4f}"
        
        return recall
        
    except AssertionError as e:
        raise AssertionError(f"Invalid input: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error computing recall: {str(e)}")
    
def precision(y_true, y_pred, average="macro"):
    """
    Calculate precision for multi-class classification
        
    Ratio of relevant retrieved instances to number of retrieved instances
    
    Args:
        y_true: Ground truth labels (1D array-like)
        y_pred: Predicted labels (1D array-like)
        average: Averaging mode, macro or micro (str)
        
    Returns:
        float: Accuracy score between 0 and 1
        
    Raises:
        AssertionError: If inputs are invalid
        ValueError: If computation fails
    """
    try:
        # Ensure numpy array
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Validate input
        assert y_true.ndim == 1, "y_true must be 1-dimensional"
        assert y_pred.ndim == 1, "y_pred must be 1-dimensional"
        assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length"
        assert len(y_true) > 0, "Input arrays cannot be empty"
        
        # Get unique classes
        classes = np.unique(np.concatenate((y_true, y_pred)))
         
        # Calculate true positives (tp) and false positives (fp) for each class
        tps = []
        fps = []        
        for cl in classes:
            tp = np.sum((y_true == cl) & (y_pred == cl))
            fp = np.sum((y_true != cl) & (y_pred == cl))
            tps.append(tp) 
            fps.append(fp)
        
        assert isinstance(average, str), "average (argument) needs to be a string"
        
        # Convert to numpy arrays
        tps = np.asarray(tps)
        fps = np.asarray(fps)
        
        if average.lower() == "macro":
            # Make sure to not divide by zero
            precisions = np.where((tps + fps) == 0, 0, tps/(tps + fps))
            precision = np.mean(precisions)
        elif average.lower() == "micro":
            retrieved = np.sum(tps) + np.sum(fps) 
            if retrieved == 0:
                precision = 0
            else:
                precision = np.sum(tps)/(retrieved)
        else:
            raise ValueError(f"average should be 'macro' or 'micro', {average} was given")       
        
        assert 0 <= precision <= 1, f"Precision score is invalid: {precision:.4f}"
        
        return precision
    
    except AssertionError as e:
        raise AssertionError(f"Invalid input: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error computing precision: {str(e)}")

def accuracy(y_true, y_pred):
    """
    Calculate accuracy for multi-class classification
        
    Ratio of correct classifications to number of classifications
    
    Args:
        y_true: Ground truth labels (1D array-like)
        y_pred: Predicted labels (1D array-like)
        
    Returns:
        float: Accuracy score between 0 and 1
        
    Raises:
        AssertionError: If inputs are invalid
        ValueError: If computation fails
    """
    try:
        # Ensure numpy array
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Validate input
        assert y_true.ndim == 1, "y_true must be 1-dimensional"
        assert y_pred.ndim == 1, "y_pred must be 1-dimensional"
        assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length"
        assert len(y_true) > 0, "Input arrays cannot be empty"
        
        # Compute accuracy
        accuracy = np.sum(y_true == y_pred)/len(y_true)
        assert 0 <= accuracy <= 1, f"Accuracy score is invalid: {accuracy:.4f}"
        
        return accuracy
        
    except AssertionError as e:
        raise AssertionError(f"Invalid input: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error computing accuracy: {str(e)}")

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
        # Ensure numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Input validation
        assert y_true.ndim == 1, "y_true must be 1-dimensional"
        assert y_pred.ndim == 1, "y_pred must be 1-dimensional"
        assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length"
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

def f1_score(y_true, y_pred):
    """
    Calculate f1 for multi-class classification.
        
    Combines precision and recall using the harmonic mean
    
    Args:
        y_true: Ground truth labels (1D array-like)
        y_pred: Predicted labels (1D array-like)
        
    Returns:
        float: F1 score between 0 and 1
        
    Raises:
        AssertionError: If inputs are invalid
        ValueError: If computation fails
    """
    try:
        rec = recall(y_true, y_pred)
        prec = precision(y_true, y_pred)
        if rec + prec != 0:
            f1 = 2*prec*rec/(rec + prec)
        else:
            f1 = 0
        
        assert 0 <= f1 <= 1, f"F1 score is invalid: {f1:.4f}"
        
        return f1
    
    except AssertionError as e:
        raise AssertionError(f"Invalid input: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error computing precision: {str(e)}")
    
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
    logits = []
    targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            logits.extend(outputs.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    
    # Convert to numpy array
    logits = np.array(logits)
    targets = np.array(targets)
    
    # Collapse logits to predictions based on largest value
    predictions = logits.argmax(axis=1)
    
    # Metrics
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    bacc = balanced_accuracy(targets, predictions)
    acc = accuracy(targets, predictions)
    prec = precision(targets, predictions)
    rec = recall(targets, predictions)
    f1 = f1_score(targets, predictions)
    
    print(f"Evaluation Results:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Balanced Accuracy: {bacc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1: {f1:.4f}")
    
    def eval_model(args):
        evaluate_model(args)