# Dennis Documentation
## Implementation tasks

1. Create a MNIST03 dataset h5py file. MNIST03 is a limited MNIST variant including only data for numbers 0-3. 
2. Custom dataset loader for MNIST03.
3. MLP with 3 hidden layers, 300 neurons in each layer, LeakyReLU activation function
4. Individual documentation

## Implementation

### Creating the dataset 

Download the original MNIST data files in gzip format

- train-images-idx3-ubyte.gz,
- train-labels-idx1-ubyte.gz,
- t10k-images-idx3-ubyte.gz,
- t10k-labels-idx1-ubyte.gz

download_prepare_mnist03.py opens the .gz data files, the train and test image data and labels 



## Challenges
### Running another person's code

The project structure is more complex than what I have experienced so far. So orienting inside the template took some time.
Also inserting my code inside the ./DJSW/main.py, ./DJSW/train.py, and ./DJSW/evaluate.py

### Another person running my code

## Tools

I have no previous experience from collaboration with code, so almost all the concepts covered in the lecture had good novelty value.
I could make use of most of the taught tools:

1. Git/Github: Had some basic knowledge about git from before but only used it sporadically and only locally.
Working on this project improved my local git workflow and working speed. Collaborating on GitHub has been a new experience.
2. Documentation with Sphinx: Adding a comprehensive documentation of own code and code from collaborators, and publish the documentation on GitHub.
3. making a proper README.md
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

