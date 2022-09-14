# Typhon Experiments
This repository contains the code of our new [**Typhon** framework](https://github.com/eXascaleInfolab/typhon) as well as the code for our experiments.

To run the code, a *Conda enviroment* must first be setup using
```
conda env create -f environment.yml
```
after cloning this repository.

One can then activate the environment using
```
conda activate typhon
```

The example experiment can finally be run using the following command:
```
python3 experiments/prostate_0.py
```
Hyperparameters and many other options can be modified in the aformentioned file.

**N.B.** that you can do a shorter run by simply adding your OS name by going into the *experiment.py* file at line 86.
