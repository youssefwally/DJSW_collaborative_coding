import pytest
from pathlib import Path
import numpy as np
import torch
from DJSW.evaluate import recall, precision, accuracy, balanced_accuracy, f1_score, evaluate_model

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
    
def test_precision():
    y_1d = np.ones(128)
    y_2d = np.ones((128, 128))
    # Check that passing y_true 2d raises errror
    with pytest.raises(AssertionError):
        precision(y_2d, y_1d, average="macro")
    # Check that passing y_pred 2d raises errror
    with pytest.raises(AssertionError):
        precision(y_1d, y_2d, average="macro")
    
    # Check that disimilar lengths raises error
    y_longer = np.ones(256)
    with pytest.raises(AssertionError):
        precision(y_longer, y_1d, average="macro")
    
    # Check that empty error raises error
    with pytest.raises(AssertionError):
        y_empty = np.array([])
        precision(y_empty, y_empty, average="macro")
    
    # Check that average = not a string raises error
    with pytest.raises(AssertionError):
        precision(y_1d, y_1d, average=-1)
    with pytest.raises(AssertionError):
        def macro():
            return True
        precision(y_1d, y_1d, average=macro)
    
    # Check that average 'macro' and 'micro' works, and that 'mid' raises error
    assert precision(y_1d, y_1d, average="macro") == 1.0
    assert precision(y_1d, y_1d, average='micro') == 1.0
    with pytest.raises(ValueError):
        precision(y_1d, y_1d, average="mid")
    
    # Check that .lower works
    assert precision(y_1d, y_1d, average="MACRO") == 1.0
    assert precision(y_longer, y_longer, average="MaCrO") == 1.0
    
    # Check that returns known results
    y_other = y_1d.copy()
    y_other[-32:] = 2
    assert precision(y_1d, y_other, average="macro") == 0.5
    assert precision(y_1d, y_other, average="micro") == 0.75

    
def test_accuracy():
    y_1d = np.ones(128)
    y_2d = np.ones((128, 128))
    # Check that passing y_true 2d raises errror
    with pytest.raises(AssertionError):
        accuracy(y_2d, y_1d)
    # Check that passing y_pred 2d raises errror
    with pytest.raises(AssertionError):
        accuracy(y_1d, y_2d)
    
    # Check that disimilar lengths raises error
    y_longer = np.ones(256)
    with pytest.raises(AssertionError):
        accuracy(y_longer, y_1d)
    
    # Check that empty error raises error
    with pytest.raises(AssertionError):
        y_empty = np.array([])
        accuracy(y_empty, y_empty)
        
    # Check that returns known results
    y_other = y_1d.copy()
    y_other[-32:] = 2
    assert accuracy(y_1d, y_other) == 0.75

def test_balanced_accuracy():
    y_1d = np.ones(128)
    y_2d = np.ones((128, 128))
    # Check that passing y_true 2d raises errror
    with pytest.raises(AssertionError):
        balanced_accuracy(y_2d, y_1d)
    # Check that passing y_pred 2d raises errror
    with pytest.raises(AssertionError):
        balanced_accuracy(y_1d, y_2d)
    
    # Check that disimilar lengths raises error
    y_longer = np.ones(256)
    with pytest.raises(AssertionError):
        balanced_accuracy(y_longer, y_1d)
    
    # Check that empty error raises error
    with pytest.raises(AssertionError):
        y_empty = np.array([])
        balanced_accuracy(y_empty, y_empty)
        
    # Check that returns known results
    y_other = y_1d.copy()
    y_other[-32:] = 2
    assert balanced_accuracy(y_1d, y_other) == 0.75
    
def test_f1_score():
    y_1d = np.ones(128)
    y_2d = np.ones((128, 128))
    # Check that passing y_true 2d raises errror
    with pytest.raises(AssertionError):
        f1_score(y_2d, y_1d)
    # Check that passing y_pred 2d raises errror
    with pytest.raises(AssertionError):
        f1_score(y_1d, y_2d)
    
    # Check that disimilar lengths raises error
    y_longer = np.ones(256)
    with pytest.raises(AssertionError):
        f1_score(y_longer, y_1d)
    
    # Check that empty array raises error
    with pytest.raises(AssertionError):
        y_empty = np.array([])
        f1_score(y_empty, y_empty)
       
    # Check that returns known results
    y_other = y_1d.copy()
    y_other[-16:] = 2
    assert f1_score(y_1d, y_other) == 0.4666666666666667 # Hard coded