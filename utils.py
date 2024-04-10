import os
import numpy as np
import torch

def save_info_0(dataset,
        hidden_num,
        learning_rate,
        epochs,
        batch_size,
        seed,
        cv_num,
        q_num,
        concept_num,
        length,
        time_spend,
        d_model,
        nhead,
        num_encoder_layers,
        dropout,
        gpu
        ,
        speed_cate,loss_rate
        ,rate1,rate2
        ,info_file):

        params_list = (
            'dataset = %s\n' % dataset,
            'learning_rate = %f\n' % learning_rate,
            'length = %d\n' % length,
            'batch_size = %d\n' % batch_size,
            'seed = %d\n' % seed,
            'q_num = %d\n' % q_num,
            'concept_num = %d\n' % concept_num,

            'length = %s\n' % length,
            'time_spend = %d\n' % time_spend,
            'd_model = %f\n' % d_model,
            'nhead = %d\n' % nhead,
            'num_encoder_layers = %d\n' % num_encoder_layers,
            'dropout = %f\n' % dropout,
            'speed_cate = %f\n' % speed_cate,
            'loss_rate = %f\n' % loss_rate ,          
            'rate1 = %f\n' % rate1  ,
            'rate2 = %f\n' % rate2           

        )
        info_file.write('%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s\n' % params_list)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, info_file,patience=7, verbose=False, delta=0, path='   .pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.info_file=info_file
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            self.info_file.write(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Valid loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.info_file.write(f'Valid loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).\n')

        self.val_loss_min = val_loss