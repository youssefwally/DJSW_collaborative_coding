# Dennis Documentation
## Implementation tasks

1. Create a MNIST03 dataset h5py file. MNIST03 is a limited MNIST variant including only data for numbers 0-3. 
2. Implement dataset loader for MNIST03.
3. Implement MLP with 3 hidden layers, 300 neurons in each layer, LeakyReLU activation function
4. Write a test script to check loaded data
5. Train DMLP on LUMI and provide the model weights
6. Evaluate MNIST49 (Sigurd's dataset) on SMLP (Sigurd's model) on LUMI
7. Prepare individual documentation

## Implementation

### Creating the dataset 

Download the following original MNIST data files in gzip format
- `train-images-idx3-ubyte.gz`
- `train-labels-idx1-ubyte.gz`
- `t10k-images-idx3-ubyte.gz`
- `t10k-labels-idx1-ubyte.gz`

and copy the files to `data/raw/MNIST_gz`. The script `datasetprep/download_prepare_mnist03.py` loads the raw train and test image data and labels from the gzip files and prepares a zipped numpy archive (`.npz`) and/or HDF5 file (`.h5`) of the reduced MNIST dataset of numbers 0-3, called MNIST03. Data and label splits in the ´.npz´ file are `X_train, y_train, X_val, y_val, X_test, y_test`. Data and label splits in the ´.h5´ file are `train, val, test`, image data is accessed through `.../images` and labels through `.../labels`.

### Implementing the MNIST03 Dataset Loader

The `Mnist03Dataset` class in `utils/dataset_mnist03_h5.py` is a dataset class for the MNIST03 dataset stored in HDF5 format. It loads the `mnist03.h5` dataset file in HDF5 format from the specified `h5_path` and returns image data and corresponding labels for the specified `split` as `torch.Tensor`. Available splits are `"train"`, `"val"` and `"test"`, image data and corresponding labels for each split are accessible through `/<split>/images` and `/<split>/labels`. The image data can be normalized to interval [0,1] through the `normalize` argument, default `True`. The image data is presented as Tensor with dimensions (N,28,28), N being the number of samples, but can be flattened to (1, 784) with using the `flatten` argument, default `True`. The `__len__` and `__getitem__` class methods ensure that the dataset class is compatible with PyTorch's Dataloader. The `_ensure_open` method makes sure that one or multiple working processes in during data loading can access the HDF5 data efficiently in parallel, and the `__del__` method ensures that the data files are closed ones the workers have concluded access from the HDF5 data file.

### Implementing DMLP

DMLP in `models/dmlp.py` is a model class for a MLP with 3 hidden layers and 300 neurons (argument `hidden_dim`, default 300) in each layer. It uses the LeakyReLU activation function, the negative slope is set with argument `negative_slope`, default is 0.01. It inherits from the PyToch `nn.Module` base class. The MLP is build as a `nn.Sequential` sequence of `nn.Linear` layer and `nn.LeakyReLU` activation function and initialized with Kaiming initialization in the `_init_weights` method. It requires input data as torch.Tensor of dimension (N, input_dim), `input_dim` is given as argument to the model. Using DMLP to train a model for MNIST03 requires the data from the dataset class to be flattened. The `forward` method returns the model output (of dimension `output_dim` given as input argument)


### Train and evaluate

To train DMLP on the MNIST03 data run:
```
# uv
uv run DJSW/main.py --username dennis --exp_name train_name --output_dir ./weights/DMLP/ --train --num_epochs 10 --batch_size 64 --lr 1e-3
# else
python DJSW/main.py --username dennis --exp_name train_name --output_dir ./weights/DMLP/ --train --num_epochs 10 --batch_size 64 --lr 1e-3
```
If the `--train` argument is set, `DJSW/main.py` calls the `train_model` function in `DJSW/train.py`. 
DMLP and the MNIST03 dataset class are available in `DJSW/train.py` through:
```python 
from models.dmlp import DMLP 
from utils.dataset_mnist03_h5 import Mnist03Dataset
```
Using the `--username` argument `"dennis"` selects the MNIST03 train and val data splits for model training and validation respectively:
```python
def train_model(args):
    ...
    elif args.username == "dennis":
        ROOT = Path(__file__).resolve().parents[1]
        H5_DIR = ROOT / "data" / "processed"
        H5_DIR.mkdir(parents=True, exist_ok=True)
        train_dataset = Mnist03Dataset(h5_path=H5_DIR / "mnist03.h5", split="train")
        val_dataset = Mnist03Dataset(h5_path=H5_DIR / "mnist03.h5", split="val")
        img_dim = 784
    ...
```
Within `train_model`, the `train_loader` and `val_loader` `DataLoader` objects are defined, taking `--batch_size` as argument for the batch size, default: 64. Moreover the function `train_pipeline` is called. For `username="dennis"`,`train_pipeline` sets `DMLP` as model:
```python
def train_pipeline(img_dim, train_loader, val_loader, args):
    ...
    elif args.username == "dennis":
        model = DMLP(input_dim=img_dim, output_dim=4, hidden_dim=300, negative_slope=0.01).to(device)
    ...
```
Training over number of epochs defined through `--num_epochs` argument, default: 10. The loss function used for training is cross entropy loss, and optimization is done with Adam:
```python
  # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
```
The optimizer takes the learning rate from the `--lr` argument, default: 1e-3. For `--num_epochs >10`, every 10th epoch a model weights checkpoint is saved in the output directory specified by the `--output_dir` argument and named according to the string given to the `--exp_name` argument

To evaluate the performance of DMLP on the MNIST03 test data run:
```
# uv
uv run DJSW/main.py --username dennis --exp_name test_name --output_dir . --load_checkpoint ./weights/DMLP/train_name_checkpoint_epoch_10.pt
# else
python DJSW/main.py --username dennis --exp_name test_name --output_dir . --load_checkpoint ./weights/DMLP/train_name_checkpoint_epoch_10.pt
```
If the `--train` argument is omitted and the `--load_checkpoint` argument is not `None`, `DJSW/main.py` calls the `eval_model` function in `DJSW/evaluate.py`. 
DMLP and the MNIST03 dataset class are available in `DJSW/evaluate.py` through:
```python 
from models.dmlp import DMLP 
from utils.dataset_mnist03_h5 import Mnist03Dataset
```
Using the `--username` argument `"dennis"` selects the MNIST03 test data splits for model evaluation, sets the model to `DMLP` and loads the model weights from the correct file specified by the `--load_checkpoint` argument:
```python
def evaluate_model(args):
    ...
    # Load test data
    ...
    elif args.username == "dennis":
        ROOT = Path(__file__).resolve().parents[1]
        H5_DIR = ROOT / "data" / "processed"
        H5_DIR.mkdir(parents=True, exist_ok=True)
        test_dataset = Mnist03Dataset(h5_path= H5_DIR / "mnist03.h5", split="test")
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        img_dim = 784
    ...

    # Load model
    model_path = args.load_checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    ...
    elif args.username == "dennis":
        model = DMLP(
            input_dim=img_dim,
            output_dim=4,
            hidden_dim=300,
            negative_slope=0.01
        )
...
```
No file is written, but the evaluation results are printed to the terminal. The evaluation results comprise mse, rmse, mae, r2, balanced accuracy, accuracy, precision, recall and f1.

### Testing 

The file `tests/test_dataset_d.py` implements two test functions for the MNIST03 dataset. The test can be run in terminal with 
```python
pytest tests/test_dataset_d.py
```
- `test_d_data_available():` checks whether the `mnist03.h5` exists in the `data/processed/` directory and outputs an error message if not
- `test_mnist03_dataset():`checks whether data type and shape of the output of `Mnist03Dataset` is correct

The test are inserted in the automatic testing workflow on GitHub

### Documentation

Documentation was set up with Sphinx and deployed to GitHub Pages. The individual reports, including this one, `dennis.md` are added under `docs/`.

## Challenges
### Running another person's code

The project structure is more complex than what I have experienced so far. So orienting inside the template took some time. After that, inserting my code inside the ./DJSW/main.py, ./DJSW/train.py, and ./DJSW/evaluate.py, a structure and code proposed and set up by Waly, was working well. Using functions with argument parser in python was new to me, but the provided example was enough to understand how this works. All in all there were surprisingly few problems when other's code. One exception was evaluation of Sigurd's model and dataset on LUMI. I probably should have tested running the code locally before moving to LUMI, but I didn't, so I found out while submitting the code to LUMI that Sigurd's data file naming was not matching what he referenced to in the evaluation script. Easily and quickly fixed by him, so I could finish the work on LUMI, but for more complex code I can imagine that troubleshooting on HPC resources becomes more difficult.

### Another person running my code

Making sure that my code fits into the proposed project structure so others could run it as is, is not something I had to be aware of before. Setting up the model and dataset loader with the given structure in mind took some time. This was the first project I worked with uv. I was not sure whether anyone else would use uv, but wanted to make sure that installation and running instructions are given not only for pip and conda, but also uv. Since uv uses a `pyproject.toml` and `uv.lock` file, instead of an environment.yaml or requirements.txt file to keep track of package dependencies, I needed to make sure to keep all up to date. When updating the installation instructions in the readme with uv, Johannes' tested the instructions, was not able to set it up, and gave good feedback so I could make it work before he accepted the pull request.
Also automatic testing and building the documentation worked without problems for my code.

## Tools

I have no previous experience from collaboration with code, so almost all the concepts covered in the lecture had good novelty value.
I could make use of most of the taught tools:

1. Git/Github: Had some basic knowledge about git from before but only used it sporadically and only locally.
Working on this project improved my local git workflow and working speed. Collaborating on GitHub has been a new experience.
2. Documentation with Sphinx: Adding a comprehensive documentation of own code and code from collaborators, and publish the documentation on GitHub.
3. Making a proper README.md
4. Testing with pytest: so far I only have been working on small scientific codebases and have not yet experienced a need for proper testing,
but see the value in collaborative projects, especially implemented with GitHub Actions, to have an automatic guard against errors caused by new commited code.
5. Lintering with Ruff
6. Writing a Dataset loader from scratch
7. Insert my code in a larger project templates, like the structure set up for this project with the Cookiecutter library.
Higher complexity than what I so far have been used to.
8. Running Code on LUMI: First time using HPC resources, I find it quite complex to set up and a lot of details to be careful and aware about.
But running the code once everything is set up was straightforward. Good opportunity to test HPC computing in a controlled environment.
9. Creating a software licence.
10. Making the repository installable.
11. I started using uv for this project, very nice alternative to pip + pyenv
12. Packing data with h5py, heard of HDF5 before but never used it, so having the HPC computing as an incentive to use it was a great opportunity.

## LUMI

On LUMI, in the home directory under `/scratch/project_projectnumber`, create your working directory. Inside this working directory, clone the DSJW_collaborative_coding project
```
git clone https://github.com/youssefwally/DJSW_collaborative_coding.git
```
To train the model `DMLP` on the MNIST03 train data, change the last line of `run.sh` to 

```
singularity exec -B ../../DJSW_env.sqsh:/user-software:image-src=/ ../../lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif python DJSW/main.py --username dennis --exp_name train_name --output_dir ./weights/DMLP/ --train 
```
and run inside the project folder 
```
sbatch run.sh
```
The slurm job id file for training DMLP on MNIST03 is `reports/DMLP_testing/slurm-14503531.out`. To evaluate Sigurd's model SMLP on his data set MNIST49 (MNIST with only numbers 4-9) change the last line of `run.sh` to 
```
singularity exec -B ../../DJSW_env.sqsh:/user-software:image-src=/ ../../lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif python DJSW/main.py --username sigurd --exp_name smlp_test_by_dennis --load_checkpoint weights/SMLP/smlp_test_1_checkpoint_epoch_10.pt --output_dir . 
```
and run 
```
sbatch run.sh
```
again. The slurm job id file for evaluating SMLP on MNIST49 is `reports/DennistestingSMLP/slurm-14508000.out`.

I'm still a bit overwhelmed by LUMI or HPC resources in general. First time running `run.sh`  I did the rookie mistake to run `main.py` directly in the LUMI terminal. Luckily I didn't get banned immediately, but only after not finding the slurm job id file I realized I forgot to actually use slurm. Submitting a job and then having to wait until the job is done withouth having verbose feedback through for example status prints is something I need to get used to.
