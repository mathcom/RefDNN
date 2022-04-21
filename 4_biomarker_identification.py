import argparse
import os
import math
import numpy as np
import pandas as pd
from datetime import datetime
from refdnn.model import REFDNN

def get_args():
    parser = argparse.ArgumentParser()
    ## positional
    parser.add_argument('inputdir', type=str, help="The output directory of the script '1_nested_cv_RefDNN.py'")
    ## optional
    parser.add_argument('-k', metavar='modelnumber', type=int, default=0, help="The fold number of outer k-fold cross-validation in '1_nested_cv_RefDNN.py'")
    parser.add_argument('-o', metavar='outputdir', type=str, default='output_4', help="A directory path for saving outputs (default:'output_4')")
    parser.add_argument('-v', metavar='verbose', type=int, default=1, help="0:No logging, 1:Basic logging to check process, 2:Full logging for debugging (default:1)")
    return parser.parse_args()
    
    
def main():
    args = get_args()
    
    inputdir = args.inputdir
    k = args.k
    outputdir = args.o
    verbose = args.v
    
    if verbose > 0:
        print('[START]')
        
    if verbose > 1:
        print('[ARGUMENT] INPUTDIR: {}'.format(args.inputdir))
        print('[ARGUMENT] K: {}'.format(args.k))
        print('[ARGUMENT] OUTPUTDIR: {}'.format(args.o))
        print('[ARGUMENT] VERBOSE: {}'.format(args.v))
        
    ## output directory
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
        
        
    ########################################################
    ## 1. Load a pretrained model
    ########################################################
    drugnames_path = os.path.join(inputdir, "checkpoint", "{:03d}_drugnames.csv".format(k))
    genenames_path = os.path.join(inputdir, "checkpoint", "{:03d}_genenames.csv".format(k))
    configs_path = os.path.join(inputdir, "checkpoint", "{:03d}_configs.csv".format(k))
    checkpoint_path = os.path.join(inputdir, "checkpoint", "{:03d}_RefDNN_cv_outer.ckpt".format(k))
    
    drugnames = pd.read_csv(drugnames_path, header=None).iloc[:,0].values
    genenames = pd.read_csv(genenames_path, header=None).iloc[:,0].values
    
    configs = pd.read_csv(configs_path, index_col=0, header=None)
    
    clf = REFDNN(hidden_units=configs.loc['HIDDEN_UNITS',1],
                 learning_rate_ftrl=configs.loc['LEARNING_RATE_FTRL',1],
                 learning_rate_adam=configs.loc['LEARNING_RATE_ADAM',1],
                 l1_regularization_strength=configs.loc['L1_REGULARIZATION_STRENGTH',1],
                 l2_regularization_strength=configs.loc['L2_REGULARIZATION_STRENGTH',1],
                 batch_size=None,
                 training_steps=None,
                 checkpoint_path=checkpoint_path)
                 
    ## time log    
    timeformat = '[TIME] [{0}] {1.year}-{1.month}-{1.day} {1.hour}:{1.minute}:{1.second}'
    if verbose > 0:
        print(timeformat.format(1, datetime.now()))

        
    ########################################################
    ## 2. Get the weights of the first dense layer
    ########################################################
    kernel_dict = clf.get_kernels(num_genes=len(genenames), num_drugs=len(drugnames))
    for layername, kernel in kernel_dict.items():
        if layername == "dense0":
            kernel_path = os.path.join(outputdir, '{:03d}_kernel.csv'.format(k))
            pd.DataFrame(data=kernel, index=genenames, columns=drugnames).to_csv(kernel_path)
            
    ## time log    
    if verbose > 0:
        print(timeformat.format(2, datetime.now()))
    
    if verbose > 0:
        print('[FINISH]')
        
        
if __name__=="__main__":
    main()