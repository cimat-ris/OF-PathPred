# OF-PathPred

This is a trajctory prediction inspired from the NextP algorithm [Liang2019] and written in tensorflow 2.

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
python3 tests/test_loo.py
```

To run with the trajnetplusplus dataset, please perform
```Python
git submodule init
```

And download the train/test datasets from: 
https://github.com/vita-epfl/trajnetplusplusdata/releases
and put them in datasets/trajnetplusplus/

A few important parameters:
* idTest gives the id in the dataset_paths array for the one dataset that is used as a test dataset, while the remaining are used for training.
* setup_loo_experiment is a function that prepares the data for training/testing. To go faster, you may set use_pickled_data=True as an argument for the preprocessing results to be stored in pickle files. The first time, obviously, you will need to set use_pickled_data=False.  
* The model is a multiple-output model. The number of output hypothesis is 2*model_parameters.output_var_dirs+1.
* model_parameters.is_mc_dropout=True allows to use MC dropout in testing.
