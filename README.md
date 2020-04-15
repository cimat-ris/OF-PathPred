# OF-PathPred

To run, install a virtual environment
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

If tensorflow 1.15.2 is not available through the command above, upgrade pip:
```Python
python3 -m pip install --upgrade pip setuptools
```
Then, you can open notebook:
```Python
.venv/bin/jupyter-notebook
```

and open the notebook located at ./keypoints/train_and_evaluate.ipynb
