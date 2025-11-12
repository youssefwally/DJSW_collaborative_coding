Johannes report
================

## Overview
As the only non-ML person on the project my tasks was mainly to implement evaluation metrics used to evaluate the others models and review the others code. In addition, as I'm developing my own Julia package, a major focus has been on implementing automated documentation and unit-testing, by using GitHub Actions.

## Evaluation metrics
The metrics implemented are `accuracy`, `precision`, `recall` and `f1_score`, with the `balanced_accuracy` being implemented by [Waly](WMLP.md).

Out of the 4 metrics, `accuracy` was definetly the simples to implement, being essentially:
```python
def accuracy(y_true, y_pred):
    ...
    accuracy = np.sum(y_true == y_pred)/len(y_true)
    return accuracy
```
However, the definition for accuracy for multi-class classificiation was the hardest to understand as it seemed complicated at first. As I'm not well versed in ML I was not sure which averaging mode is typically used when evaluating models, and therefore I added it as a `str` switch supporting both "micro" and "macro" averages. See for instance the implementation of `recall`:
```python
def recall(y_true, y_pred, average="macro"):
    """
    Calculate recall for multi-class classification.
        
    Ratio of relevant retrieved instances to number of relevant instances.
    
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
        recalls = np.where((tps + fns) == 0, 0, tps / (tps + fns))
        recall = np.mean(recalls)
    elif average.lower() == "micro":
        relevant = np.sum(tps) + np.sum(fns) 
        if relevant == 0:
            recall = 0
        else:
            recall = np.sum(tps)/(relevant)
    else:
        raise ValueError(f"average should be 'macro' or 'micro', {average} was given")       
    
    return recall
```
This is also supported for `precision` where the only difference is essentially the definition where the true negative (`tn`) is swapped with the true positive (`tp`). I also saw some mentions of the f1 score, so I wanted to implement it:
```python
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
    rec = recall(y_true, y_pred)
    prec = precision(y_true, y_pred)
    if rec + prec != 0:
        f1 = 2*prec*rec/(rec + prec)
    else:
        f1 = 0
      
    return f1
``` 

## Sphinx documentation
Choose to follow mostly the [tutorial](https://fys-8805-collaborative-coding.github.io/lecture-material/documentation/) on how to setup Sphinx, as it was new to me. I also tried to make the folder structure similar to how I have implemented it in my own Julia project, however it seemed to complicate the website paths.

When it comes to automating the documentation, the instructions in the tutorial seemed outdated and with the way our project was setup I had to manually install the dependencies:
```{code-block} yaml
name: Build documentation

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

permissions:
  contents: write

jobs:
  docs:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout
      uses: actions/checkout@v4

    - name: Set up micromamba
      uses: mamba-org/setup-micromamba@v1 #uses https://github.com/mamba-org/setup-micromamba
      with:
        create-args: >-
          python=3.12
          pip
        environment-name: docs-env
        micromamba-version: 'latest'
        generate-run-shell: false

    - name: Install dependencies
      run: |
        pip install -r docs/requirements-docs.txt
          
    - name: Sphinx build
      run: |
        sphinx-build docs docs/_build
      shell: bash -el {0}

    - name: Deploy to GitHub Pages
      uses: peaceiris/actions-gh-pages@v4
      if: ${{ github.event_name == 'push' && github.ref == 'refs/heads/main' }}
      with:
        publish_branch: gh-pages
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: docs/_build/
        force_orphan: true
```
There might have been a better way to do it but I'm new to yaml and python related GitHub Actions. 

## pytest testing
Writing tests was daunting at first, specially since I have never used *pytest*, but by writing comments about what wanted to test made it simpler to create simple states that tests some of the common mistakes in which the metrics might be called. Another important aspect was to check that the asserts works as expected such that metrics produce sensible results. Most tests looks something like this:
```python
def test_recall():
    y_1d = np.ones(128)
    y_2d = np.ones((128, 128))
    # Check that passing y_true 2d raises errror
    with pytest.raises(AssertionError):
        recall(y_2d, y_1d, average="macro")
    # Check that passing y_pred 2d raises errror
    with pytest.raises(AssertionError):
        recall(y_1d, y_2d, average="macro")

    # Check that disimilar lengths raises error
    y_longer = np.ones(256)
    with pytest.raises(AssertionError):
        recall(y_longer, y_1d, average="macro")

    # Check that empty error raises error
    with pytest.raises(AssertionError):
        y_empty = np.array([])
        recall(y_empty, y_empty, average="macro")

    # Check that average = not a string raises error
    with pytest.raises(AssertionError):
        recall(y_1d, y_1d, average=-1)
    with pytest.raises(AssertionError):
        def macro():
            return True
        recall(y_1d, y_1d, average=macro)

    # Check that average 'macro' and 'micro' works, and that 'mid' raises error
    assert recall(y_1d, y_1d, average="macro") == 1.0
    assert recall(y_1d, y_1d, average='micro') == 1.0
    with pytest.raises(ValueError):
        recall(y_1d, y_1d, average="mid")

    # Check that .lower works
    assert recall(y_1d, y_1d, average="MACRO") == 1.0
    assert recall(y_longer, y_longer, average="MaCrO") == 1.0

    # Check that returns known results
    y_other = y_1d.copy()
    y_other[-32:] = 2
    assert recall(y_1d, y_other, average="macro") == 0.375
    assert recall(y_1d, y_other, average="micro") == 0.75
```

Having previously set-up the sphinx action made setting up the automatic testing nearly trivial where the only difference was to run `pytest` instead of `sphinx-build ...` and making sure to use the correct dependency file:
```{code-block} yaml
name: Automatic testing

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout
      uses: actions/checkout@v4

    - name: Set up micromamba
      uses: mamba-org/setup-micromamba@v1 #uses https://github.com/mamba-org/setup-micromamba
      with:
        create-args: >-
          python=3.12
          pip
        environment-name: docs-env
        micromamba-version: 'latest'
        generate-run-shell: false

    - name: Install dependencies
      run: |
        pip install -r requirements.txt

    - name: Run tests
      run: |
        pytest
      shell: bash -el {0}
```

## A section of what you found easy/difficult about running another person’s code.
1. As I'm not a ML-researcher myself it is a bit daunting to know how to implement things 
I have never heard about.

## A section of what you found easy/difficult about another person running your code.

## What tools from the course did you use in the home exam that you did not know about from before.
1. I used sphinx, which I previously never h
2. LUMI
3. pytest
4. uv

## Describe your experience with running jobs on LUMI. (Include Job-IDs in description)