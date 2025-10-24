# WMLP Documentation
## WMLP
1. Implemented a WMLP model, 2 hidden layers; 100 neurons each, with LeakyReLU as the activation function.
2. Model was trained by optimizing the Cross Entropy Loss as USPS is a multi-class classification task.
3. Model was evaluated with Balanced Accuracy as a evaluation metric to be robustly evaluate the model across different classes.
## wdataloader
1. Files are saved in an h5 file to save memory and be compatible with LUMI.
2. Images are loaded into memory from disk only when needed to not use up memory.
3. made an function that returns the input dim so that it can be passed directly to the model.
## Misc
1. Used CookieCutter to make boilerplate project outlet to follow worldwide code conventions
2. -Running another persons code section-
3. --Another persons running my code section--
4. Sphinx and LUMI
5. For such a simple project it was fairly easy to run jobs on LUMI. However, I would predict that for a large project with a rapidly changing environment it will be time consuming. (Job-IDs: 13618444, 13631080)