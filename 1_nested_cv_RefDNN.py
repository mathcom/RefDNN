import argparse
import os
import math
import numpy as np
import pandas as pd
import skopt
from datetime import datetime
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from refdnn.model import REFDNN
from refdnn.dataset import DATASET


def get_args():
    parser = argparse.ArgumentParser()
    ## positional
    parser.add_argument('responseFile', type=str, help="A filepath of drug response data for TRAINING")
    parser.add_argument('expressionFile', type=str, help="A filepath of gene expression data for TRAINING")
    parser.add_argument('fingerprintFile', type=str, help="A filepath of fingerprint data for TRAINING")
    ## optional
    parser.add_argument('-o', metavar='outputdir', type=str, default='output_1', help="A directory path for saving outputs (default:'output_1')")
    parser.add_argument('-b', metavar='batchsize', type=int, default=64, help="A size of batch on training process. The small size is recommended if an available size of RAM is small (default: 64)")
    parser.add_argument('-t', metavar='numtrainingsteps', type=int, default=5000, help="Number of training steps on training process. It is recommended that the steps is larger than (numpairs / batchsize) (default: 5000)")
    parser.add_argument('-s', metavar='numbayesiansearch', type=int, default=20, help="Number of bayesian search for hyperparameter tuning (default: 20)")
    parser.add_argument('-k', metavar='outerkfold', type=int, default=5, help="K for outer k-fold cross validation (default: 5)")
    parser.add_argument('-l', metavar='innerkfold', type=int, default=3, help="L for inner l-fold cross validation (default: 3)")
    parser.add_argument('-v', metavar='verbose', type=int, default=1, help="0:No logging, 1:Basic logging to check process, 2:Full logging for debugging (default:1)")
    return parser.parse_args()
    
def main():
    args = get_args()
    
    global outputdir
    global checkpointdir
    global verbose
    
    outputdir = args.o
    verbose = args.v
    
    if verbose > 0:
        print('[START]')
    
    if verbose > 1:
        print('[ARGUMENT] RESPONSEFILE: {}'.format(args.responseFile))
        print('[ARGUMENT] EXPRESSIONFILE: {}'.format(args.expressionFile))
        print('[ARGUMENT] FINGERPRINTFILE: {}'.format(args.fingerprintFile))
        print('[ARGUMENT] OUTPUTDIR: {}'.format(args.o))
        print('[ARGUMENT] NUMBAYESIANSEARCH: {}'.format(args.s))
        print('[ARGUMENT] OUTERKFOLD: {}'.format(args.k))
        print('[ARGUMENT] INNERKFOLD: {}'.format(args.l))
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
    global dataset
    
    responseFile = args.responseFile
    expressionFile = args.expressionFile
    fingerprintFile = args.fingerprintFile
    
    dataset = DATASET(responseFile, expressionFile, fingerprintFile)
    if verbose > 0:
        print('[DATA] NUM_PAIRS: {}'.format(len(dataset)))
        print('[DATA] NUM_DRUGS: {}'.format(len(dataset.get_drugs(unique=True))))
        print('[DATA] NUM_CELLS: {}'.format(len(dataset.get_cells(unique=True))))
        print('[DATA] NUM_GENES: {}'.format(len(dataset.get_genes())))
        print('[DATA] NUM_SENSITIVITY: {}'.format(np.count_nonzero(dataset.get_labels()==0)))
        print('[DATA] NUM_RESISTANCE: {}'.format(np.count_nonzero(dataset.get_labels()==1)))
    
    ## time log
    timeformat = '[TIME] [{0}] {1.year}-{1.month}-{1.day} {1.hour}:{1.minute}:{1.second}'
    if verbose > 0:
        print(timeformat.format(1, datetime.now()))
    
    
    ########################################################
    ## 2. Define the space of hyperparameters
    ########################################################
    ## 2-1) Set the range of hyperparameters
    space_hidden_units = skopt.space.Integer(low=4, high=128, name='hidden_units')
    space_learning_rate_ftrl = skopt.space.Real(low=1e-6, high=1e-1, prior='log-uniform', name='learning_rate_ftrl')
    space_learning_rate_adam = skopt.space.Real(low=1e-6, high=1e-1, prior='log-uniform', name='learning_rate_adam')
    space_l1_regularization_strength = skopt.space.Real(low=1e-3, high=1e+2, prior='log-uniform', name='l1_regularization_strength')
    space_l2_regularization_strength = skopt.space.Real(low=1e-3, high=1e+2, prior='log-uniform', name='l2_regularization_strength')

    ## 2-2) Define hyperparmeter space
    dimensions_hyperparameters = [space_hidden_units,
                                  space_learning_rate_ftrl,
                                  space_learning_rate_adam,
                                  space_l1_regularization_strength,
                                  space_l2_regularization_strength]
    
    ## time log
    if verbose > 0:
        print(timeformat.format(2, datetime.now()))
    
    
    #######################################################
    ## 3. Start the hyperparameter tuning jobs
    ########################################################
    global fitness_step
    global fitness_idx_train
    global fitness_idx_test
    global innerkfold
    global batchsize
    global numtrainingsteps
    
    outerkfold = args.k
    innerkfold = args.l
    numbayesiansearch = args.s
    batchsize = args.b
    numtrainingsteps = args.t

    ## 3-1) init lists for metrics
    ACCURACY_outer = []
    AUCROC_outer = []
    AUCPR_outer = []

    ## 3-2) init lists for hyperparameters
    Hidden_units_outer = []
    Learning_rate_ftrl_outer = []
    Learning_rate_adam_outer = []
    L1_strength_outer = []
    L2_strength_outer = []
    
    kf = StratifiedKFold(n_splits=outerkfold, shuffle=True)
    for k, (idx_train, idx_test) in enumerate(kf.split(X=np.zeros(len(dataset)), y=dataset.get_drugs())):        
        fitness_step = 1
        fitness_idx_train = idx_train
        fitness_idx_test = idx_test
        
        ## 3-3) Bayesian optimization with gaussian process
        if verbose > 0:
            print('[OUTER] [{}/{}] NOW TUNING THE MODEL USING BAYESIAN OPTIMIZATION...'.format(k+1, kf.get_n_splits()))
            
        search_result = skopt.gp_minimize(func=fitness,
                                          dimensions=dimensions_hyperparameters,
                                          n_calls=numbayesiansearch,
                                          n_initial_points=3, # 'n_random_starts' is deprecated in skopt 0.8 and replaced by 'n_initial_points'
                                          acq_func='EI',
                                          noise=1e-10,
                                          verbose=0)
        BEST_HIDDEN_UNITS = search_result.x[0]
        BEST_LEARNING_RATE_FTRL = search_result.x[1]
        BEST_LEARNING_RATE_ADAM = search_result.x[2]
        BEST_L1_REGULARIZATION_STRENGTH = search_result.x[3]
        BEST_L2_REGULARIZATION_STRENGTH = search_result.x[4]
        BEST_TRAINING_ACCURACY = search_result.fun
        
        configs_path = os.path.join(checkpointdir, "{:03d}_configs.csv".format(k))
        with open(configs_path, 'w') as fout:
            fout.write("HIDDEN_UNITS,{:d}\n".format(BEST_HIDDEN_UNITS))
            fout.write("LEARNING_RATE_FTRL,{:.6f}\n".format(BEST_LEARNING_RATE_FTRL))
            fout.write("LEARNING_RATE_ADAM,{:.6f}\n".format(BEST_LEARNING_RATE_ADAM))
            fout.write("L1_REGULARIZATION_STRENGTH,{:.6f}\n".format(BEST_L1_REGULARIZATION_STRENGTH))
            fout.write("L2_REGULARIZATION_STRENGTH,{:.6f}\n".format(BEST_L2_REGULARIZATION_STRENGTH))
        
        Hidden_units_outer.append(BEST_HIDDEN_UNITS)
        Learning_rate_ftrl_outer.append(BEST_LEARNING_RATE_FTRL)
        Learning_rate_adam_outer.append(BEST_LEARNING_RATE_ADAM)
        L1_strength_outer.append(BEST_L1_REGULARIZATION_STRENGTH)
        L2_strength_outer.append(BEST_L2_REGULARIZATION_STRENGTH)
        
        if verbose > 0:
            print('[OUTER] [{}/{}] BEST_HIDDEN_UNITS : {}'.format(k+1, kf.get_n_splits(), BEST_HIDDEN_UNITS))
            print('[OUTER] [{}/{}] BEST_LEARNING_RATE_FTRL : {:.3e}'.format(k+1, kf.get_n_splits(), BEST_LEARNING_RATE_FTRL))
            print('[OUTER] [{}/{}] BEST_LEARNING_RATE_ADAM : {:.3e}'.format(k+1, kf.get_n_splits(), BEST_LEARNING_RATE_ADAM))
            print('[OUTER] [{}/{}] BEST_L1_REGULARIZATION_STRENGTH : {:.3e}'.format(k+1, kf.get_n_splits(), BEST_L1_REGULARIZATION_STRENGTH))
            print('[OUTER] [{}/{}] BEST_L2_REGULARIZATION_STRENGTH : {:.3e}'.format(k+1, kf.get_n_splits(), BEST_L2_REGULARIZATION_STRENGTH))
            print('[OUTER] [{}/{}] BEST_TRAINING_ACCURACY : {:.3f}'.format(k+1, kf.get_n_splits(), BEST_TRAINING_ACCURACY))
        
        ## 3-4) Dataset
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
        
        ## 3-5) Create a model using the best parameters
        if verbose > 0:
            print('[OUTER] [{}/{}] NOW TRAINING THE MODEL WITH BEST PARAMETERS...'.format(k+1, kf.get_n_splits()))
            
        checkpoint_path = os.path.join(checkpointdir, "{:03d}_RefDNN_cv_outer.ckpt".format(k))
        clf = REFDNN(hidden_units=BEST_HIDDEN_UNITS,
                     learning_rate_ftrl=BEST_LEARNING_RATE_FTRL,
                     learning_rate_adam=BEST_LEARNING_RATE_ADAM,
                     l1_regularization_strength=BEST_L1_REGULARIZATION_STRENGTH,
                     l2_regularization_strength=BEST_L2_REGULARIZATION_STRENGTH,
                     batch_size=batchsize,
                     training_steps=numtrainingsteps,
                     checkpoint_path=checkpoint_path)
                    
        ## 3-6) Fit a model
        history = clf.fit(X_train, S_train, I_train, Y_train,
                          X_valid, S_valid, I_valid, Y_valid,
                          verbose=verbose)
        
        ## 3-7) Compute the metric
        Pred_test = clf.predict(X_test, S_test, verbose=verbose)
        Prob_test = clf.predict_proba(X_test, S_test, verbose=verbose)
        
        ACCURACY_outer_k = accuracy_score(Y_test, Pred_test)
        ACCURACY_outer.append(ACCURACY_outer_k)

        AUCROC_outer_k = roc_auc_score(Y_test, Prob_test)
        AUCROC_outer.append(AUCROC_outer_k)
        
        AUCPR_outer_k = average_precision_score(Y_test, Prob_test)
        AUCPR_outer.append(AUCPR_outer_k)
        
        if verbose > 0:
            print('[OUTER] [{}/{}] BEST_TEST_ACCURACY : {:.3f}'.format(k+1, kf.get_n_splits(), ACCURACY_outer_k))
            print('[OUTER] [{}/{}] BEST_TEST_AUCROC : {:.3f}'.format(k+1, kf.get_n_splits(), AUCROC_outer_k))
            print('[OUTER] [{}/{}] BEST_TEST_AUCPR : {:.3f}'.format(k+1, kf.get_n_splits(), AUCPR_outer_k))
            
        ## 3-8) Save meta data
        drugnames_path = os.path.join(checkpointdir, "{:03d}_drugnames.csv".format(k))
        with open(drugnames_path, 'w') as fout:
            for drugname in base_drugs:
                fout.write("{}\n".format(drugname))
                
        genenames_path = os.path.join(checkpointdir, "{:03d}_genenames.csv".format(k))
        with open(genenames_path, 'w') as fout:
            for genename in dataset.get_genes():
                fout.write("{}\n".format(genename))
            
        ## time log   
        if verbose > 0:
            print(timeformat.format(3, datetime.now()))
    
    #######################################################
    ## 4. Save the results
    ########################################################
    res = pd.DataFrame.from_dict({'ACCURACY':ACCURACY_outer,
                                  'AUCROC':AUCROC_outer,
                                  'AUCPR':AUCPR_outer,
                                  'Hidden_units':Hidden_units_outer,
                                  'Learning_rate_ftrl':Learning_rate_ftrl_outer,
                                  'Learning_rate_adam':Learning_rate_adam_outer,
                                  'L1_regularization_strength':L1_strength_outer,
                                  'L2_regularization_strength':L2_strength_outer})
    res = res[['ACCURACY', 'AUCROC', 'AUCPR', 'Hidden_units', 'Learning_rate_ftrl', 'Learning_rate_adam', 'L1_regularization_strength', 'L2_regularization_strength']]
    res.to_csv(os.path.join(outputdir, 'metrics_hyperparameters.csv'), sep=',')
    
    ## time log    
    if verbose > 0:
        print(timeformat.format(4, datetime.now()))
    
    if verbose > 0:
        print('[FINISH]')
    

def fitness(hyperparameters):
    global outputdir
    global checkpointdir
    global verbose
    global dataset
    global fitness_step
    global fitness_idx_train
    global fitness_idx_test
    global innerkfold
    global batchsize
    global numtrainingsteps
    
    ## 1. Hyperparameters
    HIDDEN_UNITS = hyperparameters[0]
    LEARNING_RATE_FTRL = hyperparameters[1]
    LEARNING_RATE_ADAM = hyperparameters[2]
    L1_REGULARIZATION_STRENGTH = hyperparameters[3]
    L2_REGULARIZATION_STRENGTH = hyperparameters[4]
    
    ## 2. 2-fold Cross Validation
    if verbose > 1:
        print('[INNER] [{}/{}] NOW EVALUATING PARAMETERS IN THE INNER LOOP...'.format(fitness_step, innerkfold))
        
    objective_metrics = 0.
    kf = StratifiedKFold(n_splits=innerkfold, shuffle=True)
    for k, (idx_construction, idx_validation) in enumerate(kf.split(X=np.zeros_like(fitness_idx_train), y=dataset.get_drugs()[fitness_idx_train])):
        ## 2-1) dataset
        idx_construction = fitness_idx_train[idx_construction]
        idx_validation = fitness_idx_train[idx_validation]
        base_drugs = np.unique(dataset.get_drugs()[idx_construction])
        
        X_construction = dataset.make_xdata(idx_construction)
        S_construction = dataset.make_sdata(base_drugs, idx_construction)
        I_construction = dataset.make_idata(base_drugs, idx_construction)
        Y_construction = dataset.make_ydata(idx_construction)
        
        X_validation = dataset.make_xdata(idx_validation)
        S_validation = dataset.make_sdata(base_drugs, idx_validation)
        I_validation = dataset.make_idata(base_drugs, idx_validation)
        Y_validation = dataset.make_ydata(idx_validation)
        
        ## 2-2) Create a model
        checkpoint_path = "RefDNN_cv_inner.ckpt"
        checkpoint_path = os.path.join(checkpointdir, checkpoint_path)
        clf = REFDNN(hidden_units=HIDDEN_UNITS,
                     learning_rate_ftrl=LEARNING_RATE_FTRL,
                     learning_rate_adam=LEARNING_RATE_ADAM,
                     l1_regularization_strength=L1_REGULARIZATION_STRENGTH,
                     l2_regularization_strength=L2_REGULARIZATION_STRENGTH,
                     batch_size=batchsize,
                     training_steps=numtrainingsteps,
                     checkpoint_path=checkpoint_path)
                    
        ## 2-3) Fit a model
        history = clf.fit(X_construction, S_construction, I_construction, Y_construction,
                          X_validation, S_validation, I_validation, Y_validation,
                          verbose=verbose)
        
        ## 2-4) Compute the metric
        Pred_validation = clf.predict(X_validation, S_validation, verbose=verbose)
        objective_metrics += accuracy_score(Y_validation, Pred_validation)
    
    training_accuracy = objective_metrics / kf.get_n_splits()
    
    if verbose > 1:
        print('[INNER] [{}/{}] HIDDEN_UNITS: {}'.format(fitness_step, innerkfold, HIDDEN_UNITS))
        print('[INNER] [{}/{}] LEARNING_RATE_FTRL: {:.3e}'.format(fitness_step, innerkfold, LEARNING_RATE_FTRL))
        print('[INNER] [{}/{}] LEARNING_RATE_ADAM: {:.3e}'.format(fitness_step, innerkfold, LEARNING_RATE_ADAM))
        print('[INNER] [{}/{}] L1_REGULARIZATION_STRENGTH: {:.3e}'.format(fitness_step, innerkfold, L1_REGULARIZATION_STRENGTH))
        print('[INNER] [{}/{}] L2_REGULARIZATION_STRENGTH: {:.3e}'.format(fitness_step, innerkfold, L2_REGULARIZATION_STRENGTH))
        print('[INNER] [{}/{}] TRAINING_ACCURACY: {:.3f}'.format(fitness_step, innerkfold, training_accuracy))
    
    ## delete temporary checkpoints
    os.system("rm {}.*".format(checkpoint_path))
    
    fitness_step += 1
    return -training_accuracy


    
    
if __name__=="__main__":
    main()
    