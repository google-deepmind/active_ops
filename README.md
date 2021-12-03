# Active Offline Policy Selection

This is supporting example code for NeurIPS 2021 paper [Active Offline Policy
Selection](https://arxiv.org/abs/2106.10251) by Ksenia Konyushkova*, Yutian
Chen*, Tom Le Paine, Caglar Gulcehre, Cosmin Paduraru, Daniel J Mankowitz,
Misha Denil, Nando de Freitas.

To simulate the active offline policy selection for a set of policies, one needs
to provide a number of files. We provide the files for 76 policies on
`` cartpole_swingup `` environemnt.

1. Sampled episodic returns for all policies on a number of evalauation episodes
(`` full-reward-samples-dict.pkl ``), or a way of sampling a new episode of
evaluation upon request for any policy. The file
`` full-reward-samples-dict.pkl `` contains a dictionary that maps a policy by
its string representation to a numpy.ndarray of of shape (5000,) (number of
reward samples).

2. Off-policy evaluation score, such as fitted Q-evaluation (FQE) for all
policies (`` ope_values.pkl ``). The file `` ope_values.pkl `` contains
dictionary that maps policy info into OPE estimates. We provide FQE scores
for the policies.

3. Actions that policies take on 1000 randomly sampled states from the offline
dataset (`` actions.pkl ``). The file `` actions.pkl `` contains a dictionary
with keys `` actions`` and `` policy_keys``. `` actions`` is a list of 1000 (
number of states used to compute the kernel) elements of numpy.ndarray type of
dimensionality 76x1 (number of policies by the dimensionality of the actions).
`` policy_keys`` contains a dictionary mapping from string representation of a
policy to the index of that policy in actions.

## Installation

To set up the virtual environment, run the following commands.
From within the `active_ops` directory:

```
python3 -m venv active_ops_env
source active_ops_env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

To run the demo with colab, enable the ```jupyter_http_over_ws``` extension:

```
jupyter serverextension enable --py jupyter_http_over_ws
```

Finally, start a server:

```
jupyter notebook \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --port=8888 \
  --NotebookApp.port_retries=0
```

## Usage

To run the code refer to `` Active_ops_experiment.ipynb  `` colab notebook.
Execute blocks of code one by one to reproduce the final plot. You can modify
various parameters maked by  `` @param `` to test various baselines in modified
settings. This code loads the example of data for cartpole_environment provided
in the data folder. Using this data, we reproduce the results of Figure 14 of
the paper.

## Citing this work

```
@inproceedings{konyushkovachen2021aops,
    title = "Active Offline Policy Selection",
    author = "Ksenia Konyushkova, Yutian Chen, Tom Le Paine, Caglar Gulcehre, Cosmin Paduraru, Daniel J Mankowitz, Misha Denil, Nando de Freitas",
    booktitle = NeurIPS,
    year = 2021
}
```

## Disclaimer

This is not an official Google product.

The datasets in this work are licensed under the Creative Commons Attribution
4.0 International License. To view a copy of this license, visit
[http://creativecommons.org/licenses/by/4.0/]
(http://creativecommons.org/licenses/by/4.0/).
