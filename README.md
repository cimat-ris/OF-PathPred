# OF-PathPred

This is an adaptation of the NextP algorithm [Liang2019] in tensorflow 2.

We introduced the following modifications:
* Stacked RNN cells for encoding.
* More dropout layers.
* A new feature for encoding the spatial interactions.

First install a virtual environment
```Python
python3 -m venv .venv
```

Then, activate this virtual environment
```Python
source .venv/bin/activate
```

Install all the required dependencies. 
```Python
pip install -r requirements.txt
```

To run, use the test_loo.py script. It runs training/testing loops in a Leave-One-Out fashion.
```Python
python tests/test_loo.py
```

A few important parameters:
* idTest gives the id in the dataset_paths array for the one dataset that is used as a test dataset, while the remaining are used for training.
* setup_loo_experiment is a function that prepares the data for training/testing. To go faster, you may set use_pickled_data=True as an argument for the preprocessing results to be stored in pickle files. The first time, obviously, you will need to set use_pickled_data=False.  
* The model is a multiple-output model. The number of output hypothesis is 2*model_parameters.output_var_dirs+1.
* model_parameters.is_mc_dropout=True allows to use MC dropout in testing.
