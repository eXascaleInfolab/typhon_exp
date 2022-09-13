#######################
## EXPERIMENT CONFIG ##
#######################
import torch
import sys, os
sys.path.insert(0, os.getcwd())
from experiment import Experiment
from pathlib import Path


cfg = {
    # Get GPU number either from the terminal or from the file name
    'trg_gpu' : sys.argv[-1] if not sys.argv[-1].endswith('.py') else Path(__file__).stem.split('_')[-1],
    'trg_n_cpu' : 8, # how many CPU threads to use
    # Datasets
    'dsets' : ['Prostate', 'Brain', 'Lung'],
    'trg_dset' : 'Prostate',
    # Transfer learning type
    # Either 'sequential' or 'parallel'
    'transfer' : 'parallel',
    # Hyperparams
    'lrates' : {
        # One per each DMs
        'train' : [5e-7, 5e-7, 5e-7],
        'spec' : [8e-10, 1e-8, 1e-8],
        # Frozen is for sequential train only, when training with frozen feature extractor
        'frozen' : [1e-7, 1e-7, 1e-7],
    },
    'dropouts' : {
        # First one for the FE, following for the DMs
        'train' : [0.3, 0.3, 0.3, 0.3],
        'spec' : [0.3, 0.3, 0.3, 0.3],
        'frozen' : [0.3, 0.3, 0.3, 0.3],
    },
    'batch_size' : {
        'train' : 8,
        'spec' : 64,
    },
    # Only for training, since in specialization it trains on all batches
    'nb_batches_per_epoch' : 1,
    'epochs' : {
        'train' : 10000,
        'spec' : 300,
    },
    'architecture' : 'vgg16_v1',
    # One per each DMs
    'loss_functions' : [torch.nn.CrossEntropyLoss(), torch.nn.CrossEntropyLoss(), torch.nn.CrossEntropyLoss()],
    # One per each DMs
    'optimizers' : [torch.optim.Adam, torch.optim.Adam, torch.optim.Adam],
    # Metrics used to compare models, i.e. which one is the best
    'opt_metrics' : {
        'bootstrap' : 'auc',
        'train' : 'auc',
        'spec' : 'auc',
    },
    # Paths and filenames
    'dsets_path' : 'datasets',
    'ramdir'     : '/dev/shm', # copying data to RAM once to speed it up
    'out_path' : 'results',
    # bootstrap
    'bootstrap_size' : 200,
    # Add timestamp to avoid overwriting on the folder (e.g. if we want to repeat the same exp)
    'timestamp' : False,
    # If we want to resume the current exp (False for a new experiment)
    # Makes only sense for typhon training
    'resume' : False,
    # Experiment name
    'exp_file' : __file__,
}
#######################

if __name__ == '__main__':
    exp = Experiment(cfg)
    exp.main_run()
