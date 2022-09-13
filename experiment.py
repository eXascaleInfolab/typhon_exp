###########################################
## THIS IS THE GENERIC EXPERIMENT SCRIPT ##
###########################################
# Check folder 'experiments/' to find the actual exps
if __name__ == '__main__':
    print("You should not call this directly. Check folder `experiments`.")
    import sys
    sys.exit()


import os
import datetime
import time
from pathlib import Path
import shutil
import torch
from brutelogger import BruteLogger
import typhon
import utils


class Experiment:
    def __init__(self, cfg):
        self.cfg = cfg

        assert (not self.cfg['resume']) or (not self.cfg['timestamp']), "Cannot resume experiment with timestamp activated"
        assert (not self.cfg['transfer'] == 'sequential') or (not self.cfg['resume']), "Cannot resume training on sequential"

        self.make_paths()

        # Setup logger
        BruteLogger.save_stdout_to_file(path=self.paths['logs'])

        # Resolve CPU threads and cuda device
        torch.set_num_threads(self.cfg['trg_n_cpu'])

        if torch.cuda.is_available():
            # Need to have a GPU and to precise it at the end of the experiment file name
            # Or in the terminal after the file name
            # Assertion blocks if we cannot cast to int, i.e. last part of experiment file name is not an int
            assert isinstance(int(self.cfg['trg_gpu']), int), "Please precise your GPU at the end of the experiment file name"
            # Will anyway stop if the index is not available or wrong
            self.cuda_device = f"cuda:{int(self.cfg['trg_gpu'])}"
        # Otherwise just go with CPU
        else:
            self.cuda_device = 'cpu'
            torch.set_num_threads(self.cfg['trg_n_cpu'])

        # Give a dropout, learning rate, optimizer and loss function specific to each DMs
        self.dropouts = {}
        self.lrates = {}
        for type in self.cfg['dropouts'].keys():
            self.dropouts[type] =  [self.cfg['dropouts'][type][0], {name:dropout for name, dropout in zip(self.cfg['dsets'], self.cfg['dropouts'][type][1:])}]
        for type in self.cfg['lrates'].keys():
            self.lrates[type] = {name:lrate for name, lrate in zip(self.cfg['dsets'], self.cfg['lrates'][type])}
        self.optimizers = {name:optim for name, optim in zip(self.cfg['dsets'], self.cfg['optimizers'])}
        self.loss_functions = {name:fct for name, fct in zip(self.cfg['dsets'], self.cfg['loss_functions'])}

        self.train_args = {
            'paths' : self.paths,
            'dsets_names' : self.cfg['dsets'],
            'architecture' : self.cfg['architecture'],
            'bootstrap_size' : self.cfg['bootstrap_size'],
            'nb_batches_per_epoch' : self.cfg['nb_batches_per_epoch'],
            'nb_epochs' : self.cfg['epochs'],
            'lrates' : self.lrates,
            'dropouts' : self.dropouts,
            'batch_size' : self.cfg['batch_size'],
            'loss_functions' : self.loss_functions,
            'optim_class' : self.optimizers,
            'opt_metrics' : self.cfg['opt_metrics'],
            'cuda_device' : self.cuda_device,
            'resume' : self.cfg['resume'],
        }

        print(f"> Config loaded successfully for {self.cfg['transfer']} training:")
        # Print the config so it is written in the log file as well
        for key, value in self.train_args.items():
            if key == 'paths': continue
            print(f">> {key}: {value}")


    def make_paths(self):
        # Local level/debug config: shorter runs
        # Simply add your `os.uname().nodename` to the list.
        is_local_run = os.uname().nodename in ['example_os_name']
        if is_local_run:
            self.cfg.update({
                'nb_batches_per_epoch' : 1,
                'epochs' : {
                    'train' : 10,
                    'spec' : 10,
                },
                # Paths and filenames
                'dsets_path' : 'datasets/tiny',
                'bootstrap_size' : 10,
            })

        # Make Path objects
        self.cfg.update({
            'dsets_path' : Path(self.cfg['dsets_path']),
            'ramdir' : Path(self.cfg['ramdir']),
            'out_path' : Path(self.cfg['out_path']),
            'exp_file' : Path(self.cfg['exp_file']),
        })

        # Copy dataset to ram for optimization
        # The slash operator '/' in the pathlib module is similar to os.path.join()
        dsets_path_ram = self.cfg['ramdir'] / self.cfg['dsets_path']
        if not is_local_run and not dsets_path_ram.is_dir():
            import shutil
            shutil.copytree(self.cfg['dsets_path'], dsets_path_ram)

        # All paths in one place
        if self.cfg['timestamp']:
            # Add timestamp in folder name to avoid duplicates
            experiment_path = self.cfg['out_path'] / f"{self.cfg['exp_file'].stem}_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
        else:
            experiment_path = self.cfg['out_path'] / f"{self.cfg['exp_file'].stem}"

        assert (not self.cfg['resume']) or experiment_path.is_dir(), ("Folder experiment does not exist, "
            "either run experiment from the beginning or remove timestamp from folder name")

        models_path = experiment_path / 'models'
        self.paths = {
            'experiment' : experiment_path,
            # Brutelogger logs
            'logs' : experiment_path / 'run_logs',
            'dsets' : {d: self.cfg['dsets_path'] / f"{d}" for d in self.cfg['dsets']}
                            if is_local_run else {d: dsets_path_ram / f"{d}" for d in self.cfg['dsets']},
            # Trained model (no specialization)
            # p for parallel and s for sequential
            'train_model_p' : models_path / 'train_model_p.pth',
            'train_model_s' : models_path / 'train_model_s.pth',
            # Model saved after the "normal training" in hydra
            'gen_model_s' : models_path / 'gen_model_s.pth',
            # Specialized models
            'spec_models_p' : {d: models_path / f"spec_model_{d}_p.pth" for d in self.cfg['dsets']},
            'spec_models_s' : {d: models_path / f"spec_model_{d}_s.pth" for d in self.cfg['dsets']},
            # bootstrap model
            'bootstrap_model' : models_path / 'bootstrap_model.pth',
            # Plots
            'metrics' : experiment_path / 'run_plot',
        }

        # Create directories
        self.paths['metrics'].mkdir(parents=True, exist_ok=True)
        self.paths['logs'].mkdir(parents=True, exist_ok=True)
        models_path.mkdir(parents=True, exist_ok=True)


    def main_run(self):
        start = time.perf_counter()
        # Need this for sequential learning
        assert self.cfg['trg_dset'] == self.cfg['dsets'][0], "Target dataset must be in first position"
        assert self.cfg['transfer'] in ['sequential', 'parallel'], "Please transfer argument must be 'sequential' or 'parallel'"
        # Copy the experiment.py and exp cfg file in the experiment dir
        shutil.copy2(self.cfg['exp_file'], self.paths['experiment'])
        shutil.copy2('experiment.py', self.paths['experiment'])

        self.typhon = typhon.Typhon(**self.train_args)
        # Ensure bootstrap is initialized
        if not self.paths['bootstrap_model'].is_file():
            print("> Bootstrap initialization missing:", self.paths['bootstrap_model'])
            self.typhon.bootstrap()
        if self.cfg['transfer'] == 'sequential':
            self.typhon.s_train(self.paths['bootstrap_model'])
            self.typhon.s_specialization(self.paths['train_model_s'])
        if self.cfg['transfer'] == 'parallel':
            if self.cfg['resume']:
                self.typhon.p_train(self.paths['train_model_p'])
            else:
                self.typhon.p_train(self.paths['bootstrap_model'])
            self.typhon.p_specialization(self.paths['train_model_p'])

        stop = time.perf_counter()
        total_time = stop - start
        print(f"Experiment ended in {int(total_time / 3600)} hours {int((total_time % 3600) / 60)} minutes {total_time % 60:.1f} seconds")
        utils.print_time('END EXPERIMENT')
