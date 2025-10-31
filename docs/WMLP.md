WMLP Documentation
================

## Overview
---

A dataset pipeline was developed for the USPS digits 0–6, including loading, and preprocessing. The processed data would be fed into a classification network using a simple MLP with two hidden layers of 100 neurons each and a Leaky ReLU activation. Evaluation was conducted using the balanced accuracy metric, which was implemented from scratch.


## WMLP

1) Implemented a WMLP model, 2 hidden layers; 100 neurons each, with LeakyReLU as the activation function.
2) Model was trained by optimizing the Cross Entropy Loss as USPS is a multi-class classification task.
3) Model was evaluated with Balanced Accuracy as a evaluation metric to be robustly evaluate the model across different classes.


### Structure

```python
class WMLP(nn.Module):
    """
    Multi-Layer Perceptron with 2 hidden layers (100 neurons each).
    
    Args:
        input_dim (int): Dimension of input features
        output_dim (int): Dimension of output
        hidden_dim (int): Number of neurons in each hidden layer (default: 100)
        negative_slope (float): Controls the angle of the negative slope for LeakyReLU (default: 0.01)
    """
    
    def __init__(self, input_dim, output_dim, hidden_dim=100, negative_slope=0.01):
        super(WMLP, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)
```
## USPS
---

The dataloader provides image loading and preprocessing for a subset of the USPS collection covering digits 0–6. Data are obtained from official sources, transformed into ready-to-use image/label pairs, partitioned, and saved in the HDF5 file format for convenient access. The loader retrieves one example at a time, ensuring only a single sample is held in memory to minimize resource usage.

## Balanced Accuracy
---

```python
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
    .........
        
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
```


## Misc
---

1) Used CookieCutter to make boilerplate project outlet to follow worldwide code conventions
2) Everything went smoothly (Dennis)
3) Everything went smoothly (one typo as error)
4) New knowledge: Sphinx and LUMI
5) For such a simple project it was fairly easy to run jobs on LUMI. However, I would predict that for a large project with a rapidly changing environment it will be time consuming. (Job-IDs: 13618444, 13631080, 14077470)

