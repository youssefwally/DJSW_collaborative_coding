import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
# -----------------------------------------------------
#Imports

import os
import sys
import traceback
import configargparse

import torch
import numpy as np
import random
from evaluate import evaluate_model
from train import train_model
# -----------------------------------------------------
#Functions

def getArguments():
    """ Parses command-line options. """
    parser = configargparse.ArgumentParser(description='Training and Evaluation Script', add_help=True)

    # User setting
    parser.add_argument('--username', default=None, type=str, required=True, 
                        choices=["dennis", "johannes", "sigurd", "waly"], help = "Select a User.")

    # Output settings
    parser.add_argument('--exp_name', default=None, type=str, required=True, 
                        help="Name of the experiment.")
    parser.add_argument('--output_dir', default=None, type=str, required=True,
                        help = "Path for output files (relative to working directory).")
    
    # General settings
    parser.add_argument('--seed', default=42, type=int, 
                        help="Set seed for deterministic training.")
    parser.add_argument('--train', action='store_true',
                        help = "If the model should be trained, otherwise evaluated.")
    parser.add_argument('--load_checkpoint', default=None, type=str, 
                        help = "Path to model checkpoint (weights, optimizer, epoch).")
                        
    # General training hyperparameters
    parser.add_argument('--num_epochs', default=10, type=int, 
                        help="Number of training epochs.")
    parser.add_argument('--batch_size', default=64, type=int, 
                        help="Training batch size.")
    parser.add_argument('--lr', default=1e-3, type=float, 
                        help="Training learning rate.")
    
    args = parser.parse_args()

    return args 
# -----------------------------------------------------
def main(args):
    if args.train:
        train_model(args)
    else:
        if args.load_checkpoint is None:
            raise ValueError("For evaluation, a model checkpoint must be provided using --load_checkpoint.")
        evaluate_model(args)
# -----------------------------------------------------
if __name__ == '__main__':
    args = getArguments()

    # Set the seed for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Raise an error if the output directory does not exist
    if not os.path.exists(args.output_dir):
        raise FileNotFoundError(f"The specified output directory '{args.output_dir}' does not exist.")

    try:
        main(args)
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        filename, line, func, text = tb[-1]  # Get the last traceback entry
        raise RuntimeError(f"Error during training: {e} (File \"{filename}\", line {line}, in {func})") from e
