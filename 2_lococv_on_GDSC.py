import argparse
import os
import math
import numpy as np
import pandas as pd
import skopt
from datetime import datetime
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from refdnn.model import REFDNN
from refdnn.dataset import DATASET, read_cancertypeFile


def get_args():
    parser = argparse.ArgumentParser()
    ## positional
    parser.add_argument('responseFile', type=str, help="A filepath of drug response data for TRAINING")
    parser.add_argument('expressionFile', type=str, help="A filepath of gene expression data for TRAINING")
    parser.add_argument('fingerprintFile', type=str, help="A filepath of fingerprint data for TRAINING")
    parser.add_argument('annotationFile', type=str, help="A filepath of cellline annotation data for TRAINING")
    ## optional
    parser.add_argument('-o', metavar='outputdir', type=str, default='output_2', help="A directory path for saving outputs (default:'output_2')")
    parser.add_argument('-b', metavar='batchsize', type=int, default=64, help="A size of batch on training process. The small size is recommended if an available size of RAM is small (default: 64)")
    parser.add_argument('-t', metavar='numtrainingsteps', type=int, default=5000, help="Number of training steps on training process. It is recommended that the steps is larger than (numpairs / batchsize) (default: 5000)")
    parser.add_argument('-v', metavar='verbose', type=int, default=1, help="0:No logging, 1:Basic logging to check process, 2:Full logging for debugging (default:1)")
    return parser.parse_args()
    
def main():
    args = get_args()
    outputdir = args.o
    verbose = args.v
    
    if verbose > 0:
        print('[START]')
    
    if verbose > 1:
        print('[ARGUMENT] RESPONSEFILE: {}'.format(args.responseFile))
        print('[ARGUMENT] EXPRESSIONFILE: {}'.format(args.expressionFile))
        print('[ARGUMENT] FINGERPRINTFILE: {}'.format(args.fingerprintFile))
        print('[ARGUMENT] OUTPUTDIR: {}'.format(args.o))
        print('[ARGUMENT] VERBOSE: {}'.format(args.v))
    
    ## output directory
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    checkpointdir = os.path.join(outputdir, "checkpoint")
    if not os.path.exists(checkpointdir):
        os.mkdir(checkpointdir)
    
    ########################################################
    ## 1. Read data
    ########################################################    
    responseFile = args.responseFile
    expressionFile = args.expressionFile
    fingerprintFile = args.fingerprintFile
    annotationFile = args.annotationFile
    
    dataset = DATASET(responseFile, expressionFile, fingerprintFile)
    if verbose > 0:
        print('[DATA] NUM_PAIRS: {}'.format(len(dataset)))
        print('[DATA] NUM_DRUGS: {}'.format(len(dataset.get_drugs(unique=True))))
        print('[DATA] NUM_CELLS: {}'.format(len(dataset.get_cells(unique=True))))
        print('[DATA] NUM_GENES: {}'.format(len(dataset.get_genes())))
        print('[DATA] NUM_SENSITIVITY: {}'.format(np.count_nonzero(dataset.get_labels()==0)))
        print('[DATA] NUM_RESISTANCE: {}'.format(np.count_nonzero(dataset.get_labels()==1)))
    
    cell2cancer = read_cancertypeFile(annotationFile)
    cancertypes = [cell2cancer.get(cell, 'others') for cell in dataset.get_cells()]
    
    ## time log
    timeformat = '[TIME] [{0}] {1.year}-{1.month}-{1.day} {1.hour}:{1.minute}:{1.second}'
    if verbose > 0:
        print(timeformat.format(1, datetime.now()))
    

    #######################################################
    ## 2. Train RefDNN using the best hyperparameters
    ########################################################
    batchsize = args.b
    numtrainingsteps = args.t
    
    ## 2-1) init lists for metrics
    ACCURACY_outer = []
    AUCROC_outer = []
    AUCPR_outer = []
    CANCER_outer = []
        
    kf = LeaveOneGroupOut()
    n_splits = kf.get_n_splits(groups=cancertypes)
    print("LeaveOneGroupOut.get_n_splits: {}".format(n_splits))
    for k, (idx_train, idx_test) in enumerate(kf.split(X=np.zeros(len(dataset)), groups=cancertypes)):        
        ## 2-2) Check a cancer type in test
        test_cancertype = np.unique([cancertypes[x] for x in idx_test])[0]
        CANCER_outer.append(test_cancertype)
        print('[{}/{}] TEST_CANCER: {}'.format(k+1, n_splits, test_cancertype))
        
        ## 2-3) Set the best values of hyperparameters
        BEST_HIDDEN_UNITS = 49
        BEST_LEARNING_RATE_FTRL = 7.94581095185585e-06
        BEST_LEARNING_RATE_ADAM = 0.0004067851789088527
        BEST_L1_REGULARIZATION_STRENGTH = 0.001
        BEST_L2_REGULARIZATION_STRENGTH = 66.7516541409175
        
        ## 2-4) Dataset
        idx_train_train, idx_train_valid = train_test_split(idx_train, test_size=0.2, stratify=dataset.get_drugs()[idx_train])
        base_drugs = np.unique(dataset.get_drugs()[idx_train_train])
        
        X_train = dataset.make_xdata(idx_train_train)
        S_train = dataset.make_sdata(base_drugs, idx_train_train)
        I_train = dataset.make_idata(base_drugs, idx_train_train)
        Y_train = dataset.make_ydata(idx_train_train)
        
        X_valid = dataset.make_xdata(idx_train_valid)
        S_valid = dataset.make_sdata(base_drugs, idx_train_valid)
        I_valid = dataset.make_idata(base_drugs, idx_train_valid)
        Y_valid = dataset.make_ydata(idx_train_valid)
        
        X_test = dataset.make_xdata(idx_test)
        S_test = dataset.make_sdata(base_drugs, idx_test)
        Y_test = dataset.make_ydata(idx_test)
        
        ## 2-5) Create a model using the best parameters
        if verbose > 0:
            print('[{}/{}] NOW TRAINING THE MODEL WITH BEST PARAMETERS...'.format(k+1, n_splits))
            
        checkpoint_path = "RefDNN_lococv_{}.ckpt".format(test_cancertype)
        checkpoint_path = os.path.join(checkpointdir, checkpoint_path)
        clf = REFDNN(hidden_units=BEST_HIDDEN_UNITS,
                     learning_rate_ftrl=BEST_LEARNING_RATE_FTRL,
                     learning_rate_adam=BEST_LEARNING_RATE_ADAM,
                     l1_regularization_strength=BEST_L1_REGULARIZATION_STRENGTH,
                     l2_regularization_strength=BEST_L2_REGULARIZATION_STRENGTH,
                     batch_size=batchsize,
                     training_steps=numtrainingsteps,
                     checkpoint_path=checkpoint_path)
                    
        ## 2-6) Fit a model
        history = clf.fit(X_train, S_train, I_train, Y_train,
                          X_valid, S_valid, I_valid, Y_valid,
                          verbose=verbose)
        
        ## 2-7) Compute the metric
        Pred_test = clf.predict(X_test, S_test, verbose=verbose)
        Prob_test = clf.predict_proba(X_test, S_test, verbose=verbose)
        
        ACCURACY_outer_k = accuracy_score(Y_test, Pred_test)
        ACCURACY_outer.append(ACCURACY_outer_k)

        AUCROC_outer_k = roc_auc_score(Y_test, Prob_test)
        AUCROC_outer.append(AUCROC_outer_k)
        
        AUCPR_outer_k = average_precision_score(Y_test, Prob_test)
        AUCPR_outer.append(AUCPR_outer_k)
        
        if verbose > 0:
            print('[{}/{}] BEST_TEST_ACCURACY : {:.3f}'.format(k+1, n_splits, ACCURACY_outer_k))
            print('[{}/{}] BEST_TEST_AUCROC : {:.3f}'.format(k+1, n_splits, AUCROC_outer_k))
            print('[{}/{}] BEST_TEST_AUCPR : {:.3f}'.format(k+1, n_splits, AUCPR_outer_k))
            
        ## time log   
        if verbose > 0:
            print(timeformat.format(3, datetime.now()))
    
    #######################################################
    ## 3. Save the results
    ########################################################
    res = pd.DataFrame.from_dict({'CANCERTYPE':CANCER_outer,
                                  'ACCURACY':ACCURACY_outer,
                                  'AUCROC':AUCROC_outer,
                                  'AUCPR':AUCPR_outer,
                                  'Hidden_units':BEST_HIDDEN_UNITS,
                                  'Learning_rate_ftrl':BEST_LEARNING_RATE_FTRL,
                                  'Learning_rate_adam':BEST_LEARNING_RATE_ADAM,
                                  'L1_regularization_strength':BEST_L1_REGULARIZATION_STRENGTH,
                                  'L2_regularization_strength':BEST_L2_REGULARIZATION_STRENGTH})
    res = res[['CANCERTYPE', 'ACCURACY', 'AUCROC', 'AUCPR', 'Hidden_units', 'Learning_rate_ftrl', 'Learning_rate_adam', 'L1_regularization_strength', 'L2_regularization_strength']]
    res.to_csv(os.path.join(outputdir, 'metrics_hyperparameters.csv'), sep=',')
    
    ## time log    
    if verbose > 0:
        print(timeformat.format(4, datetime.now()))
    
    if verbose > 0:
        print('[FINISH]')

    
    
if __name__=="__main__":
    main()
    