# Sigurd — Individual task

## Task Overview
My task, in addition to the overall project goals, consisted of the following sub-tasks:

- Implement a script to download the MNIST dataset and write it to HDF5, and a dataloader to load only the images of the digits 4–9.
- Implement SMLP: a multi-layer perceptron with 4 hidden layers, each with 77 neurons, using ReLU activations.
- Train the SMLP model on the LUMI supercomputer and save checkpoints.
- Evaluate Wally's WMLP using the provided evaluation script (also on LUMI) and produce comparative metrics.

## Network implementation — SMLP

Files: `models/smlp.py`

Design summary:

- Architecture: input layer -> 4 hidden layers (each 77 units, ReLU) -> output layer. Input size is 28*28=784 (flattened MNIST), output size equals number of classes (6 for digits 4–9).
- Forward pass: flatten input; pass through dense layers with ReLU; final linear layer returns logits (no softmax).
- Loss & optimizer: CrossEntropyLoss + Adam. These are standard choices for all the MLP experiments.


## MNIST dataset & HDF5 writer (in-depth)

Files: `datasetprep/create_mnist_dataset.py`, `utils/mnist_dataset.py` (`MnistH5Dataset`)

What the writer does:

- Downloads MNIST using `torchvision.datasets.MNIST` into `data/raw/mnist/`.
- Writes two datasets into an HDF5 file: `images` (shape (N,28,28)) and `labels` (shape (N,)).
- Using the `labels_to_keep` argument you can create reduced HDF5 subset of MNIST containing only images with specific labels.

What the dataset loader does:

- `MnistH5Dataset` opens the HDF5 file and returns (image_tensor, label). Images are converted to float tensors in [0,1].
- `target_transform` is supported — the codebase uses `lambda t: int(t)-4` so labels 4-9 map to 0-5 during training/evaluation without altering the stored files.


## Script & function reference

This section documents the primary functions/classes and CLI flags so teammates can quickly reuse them.

- datasetprep/create_mnist_dataset.py
	- `create_mnist_h5`(output_path: str | Path = "data/processed/mnist.h5", train: bool = True, download: bool = True, labels_to_keep: list[int] | None = None) -> Path
		- `output_path`: path to write the .h5 file (parent dir is created if missing).
		- `train`: True -> download/write the training split; False -> write the test split.
		- `download`: allow torchvision to download MNIST if missing.
		- `labels_to_keep`: optional list of integer labels to keep (e.g. [4,5,6,7,8,9]). If provided the function filters images/labels before writing.
		- Returns the Path to the written HDF5 file.
		
- utils/mnist_dataset.py
	- class `MnistH5Dataset`(torch.utils.data.Dataset)
		- `__init__`(h5_path: str | Path, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None)
			- `h5_path`: path to an HDF5 file containing `images` and `labels` datasets.
			- `transform`: optional transform applied to the image tensor (after scaling to [0,1]).
			- `target_transform`: optional transform applied to the label (used to remap 4..9 -> 0..5 for SMLP).
		- `__len__()`: returns number of samples (reads labels shape lazily).
		- `__getitem__`(idx): returns (img_tensor, label). Image is a float tensor in [0,1] with shape (C,H,W).

- models/smlp.py
	- class `SMLP`(nn.Module)
		- `__init__`(input_size=784, hidden_size=77, output_size=6)
			- `input_size`: flattened image dimensionality (28*28=784)
			- `hidden_size`: number of neurons per hidden layer (77 used in the project)
			- `output_size`: number of classes (6 for digits 4..9 remapped to 0..5)
		- `forward(x)`: expects x shaped (B, input_size) and returns logits (B, output_size)

- DJSW/main.py
	- Important CLI flags:
		- `--username` <username> (required) : selects the user's pipeline (`sigurd`, `waly`, `dennis`).
		- `--exp_name` <name> (required) : experiment name used in checkpoints.
		- `--output_dir` <path> (required) : directory for outputs and checkpoints (must exist).
		- `--train` (flag) : run training when present; otherwise evaluation (requires --load_checkpoint).
		- `--load_checkpoint` <path> : checkpoint path for evaluation.
		- `--num_epochs`, `--batch_size`, `--lr` : training hyperparameters.

- DJSW/train.py
	- `train_model`(args): when `args.username` == 'sigurd' the pipeline:
		1. Loads `data/processed/mnist_4_9.h5` with MnistH5Dataset(target_transform=lambda t: int(t)-4).
		2. Splits into train/val (90/10) via random_split.
		3. Creates DataLoaders and calls `train_pipeline` which flattens inputs and runs epochs training the model.
		4. Checkpoints saved to args dependent directory: `{args.output_dir}/{args.exp_name}_checkpoint_epoch_{N}.pt`.

- DJSW/evaluate.py 
	- `evaluate_model`(args): requires args.load_checkpoint; when `args.username` == 'sigurd' it:
		1. Loads `data/processed/mnist_4_9.h5` via MnistH5Dataset with the same target_transform used in training.
		2. Loads the checkpoint with `torch.load(...)` and `model.load_state_dict(...)` (if checkpoint wraps the state dict, use the robust snippet in the appendix).
		3. Flattens inputs before inference on test set.
  		4. Computes metrics (accuracy, precision, recall, F1, balanced accuracy, etc.).

## Accuracy & metrics (in-depth)

Approach and implementation:

- The evaluation code computes a variety of metrics: accuracy, precision, recall, F1, balanced accuracy, MSE/RMSE/MAE/R² when appropriate. These functions take inn the output predictions and labels, and output the respective measurement.  

## Running someone else's code

The shared infrastructure (`DJSW/main.py`) and agreed wrapper structure for models/datasets simplified made running both my own and other models a breeze. Running on LUMI took a bit of time to set up, but it was mostly due to going through the course on windows and then switching to mac for the home exam.

I also spent a bit of time debugging why pointing to folders in a certain way worked for other people and not for me, where I found that while someone starting the path with "../" worked for them, but I had to change it to "./" to make it work for me. 

## Someone else running my code
No one has ran my code yet so it is difficult to say, but I assume it will be as easy for them as it was for me. 

What might have been done differently. 
- We've run the code on LUMI using the `run.sh` file, adapting it for each run. Using only a single file reduces clutter, but keeping a separate one for each LUMI run might have allowed more reproducibility.  

## Tools used

- PyTorch + torchvision — downloading the dataset and base functions for the MLP.
- h5py / HDF5 — dataset storage.
- Sphinx — documentation generation.
- PyTest — small unit tests for loader/writer.
- uv — While only used locally for prototyping, the course introduced me to uv as an alternative to conda. It is great! Much appreciated. 

## LUMI experience (job evidence)
Running on LUMI was an interesting experience. I learned a lot from it, not just LUMI specific stuff, but also got more comfortable with the linux terminal, which is also really nice. 

The following files are the job-files produced from training and evaluating on LUMI: 
- Job slurm-14079744.out — 10 epoch training run. Checkpoint saved to `weights/SMLP/`.
- Job slurm-14362915.out - Evaluation run on WMLP.

## Artifacts and repo pointers

- `data/processed/mnist_4_9.h5` (training HDF5)
- `data/processed/mnist_test_9_5.h5` (test HDF5)
- `weights/SMLP/` — saved model checkpoint from training
- `reports/` — Slurm logs and run artifacts

