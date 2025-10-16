import numpy as np

import torch
from models.wmlp import WMLP
from torch.utils.data import DataLoader
from utils.wdataloader import USPS06Dataset
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def evaluate_model(args):
    """Evaluate a WMLP model on test data."""
    model_path = args.load_checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data
    test_dataset = USPS06Dataset(set_type="test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    img_dim = test_dataset.get_input_dim()

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    print(checkpoint.keys())
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
    
    print(f"Evaluation Results:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")


def eval_model(args):
    evaluate_model(args)