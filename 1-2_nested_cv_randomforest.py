import argparse
import os
import math
import numpy as np
import pandas as pd
import skopt
from datetime import datetime
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from refdnn.dataset import DATASET


def get_args():
    parser = argparse.ArgumentParser()
    ## positional
    parser.add_argument('responseFile', type=str, help="A filepath of drug response data for TRAINING")
    parser.add_argument('expressionFile', type=str, help="A filepath of gene expression data for TRAINING")
    parser.add_argument('fingerprintFile', type=str, help="A filepath of fingerprint data for TRAINING")
    ## optional
    parser.add_argument('-o', metavar='outputdir', type=str, default='output_1-2', help="A directory path for saving outputs (default:'output_1-2')")
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
    space_n_estimators = skopt.space.Integer(low=10, high=100, name='n_estimators')
    space_max_depth = skopt.space.Integer(low=3, high=10, name='max_depth')
    ## 2-2) Define hyperparmeter space
    dimensions_hyperparameters = [space_n_estimators,
                                  space_max_depth]
    
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
    
    outerkfold = args.k
    innerkfold = args.l
    numbayesiansearch = args.s
    
    ## 3-1) init lists for metrics
    ACCURACY_outer = []
    AUCROC_outer = []
    AUCPR_outer = []
    ## 3-2) init lists for hyperparameters
    N_estimators_outer = []
    Max_depth_outer = []
    
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
        BEST_N_ESTIMTORS = search_result.x[0]
        BEST_MAX_DEPTH = search_result.x[1]
        BEST_TRAINING_ACCURACY = search_result.fun
        
        N_estimators_outer.append(BEST_N_ESTIMTORS)
        Max_depth_outer.append(BEST_MAX_DEPTH)
        
        if verbose > 0:
            print('[OUTER] [{}/{}] BEST_N_ESTIMTORS : {:d}'.format(k+1, kf.get_n_splits(), BEST_N_ESTIMTORS))
            print('[OUTER] [{}/{}] BEST_MAX_DEPTH : {:d}'.format(k+1, kf.get_n_splits(), BEST_MAX_DEPTH))
            print('[OUTER] [{}/{}] BEST_TRAINING_ACCURACY : {:.3f}'.format(k+1, kf.get_n_splits(), BEST_TRAINING_ACCURACY))
        
        ## 3-4) Dataset
        idx_train_train, idx_train_valid = train_test_split(idx_train, test_size=0.2, stratify=dataset.get_drugs()[idx_train])
        base_drugs = np.unique(dataset.get_drugs()[idx_train_train])
        
        X_train = dataset.make_xdata(idx_train_train)
        Y_train = dataset.make_ydata(idx_train_train).ravel()
        
        X_valid = dataset.make_xdata(idx_train_valid)
        Y_valid = dataset.make_ydata(idx_train_valid).ravel()
        
        X_test = dataset.make_xdata(idx_test)
        Y_test = dataset.make_ydata(idx_test).ravel()
        
        ## 3-5) Create a model using the best parameters
        if verbose > 0:
            print('[OUTER] [{}/{}] NOW TRAINING THE MODEL WITH BEST PARAMETERS...'.format(k+1, kf.get_n_splits()))
            
        clf = RandomForestClassifier(n_estimators=BEST_N_ESTIMTORS,
                                     max_depth=BEST_MAX_DEPTH,
                                     n_jobs=None)
                    
        ## 3-6) Fit a model
        history = clf.fit(X_train, Y_train)
        
        ## 3-7) Compute the metric
        Pred_test = clf.predict(X_test)
        Prob_test = clf.predict_proba(X_test)[:,1]
        
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
            
        ## time log   
        if verbose > 0:
            print(timeformat.format(3, datetime.now()))
    
    #######################################################
    ## 4. Save the results
    ########################################################
    res = pd.DataFrame.from_dict({'ACCURACY':ACCURACY_outer,
                                  'AUCROC':AUCROC_outer,
                                  'AUCPR':AUCPR_outer,
                                  'N_ESTIMTORS':N_estimators_outer,
                                  'MAX_DEPTH':Max_depth_outer})
    res = res[['ACCURACY', 'AUCROC', 'AUCPR', 'N_ESTIMTORS', 'MAX_DEPTH']]
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
    
    ## 1. Hyperparameters
    N_ESTIMTORS = hyperparameters[0]
    MAX_DEPTH = hyperparameters[1]
    
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
        Y_construction = dataset.make_ydata(idx_construction).ravel()
        
        X_validation = dataset.make_xdata(idx_validation)
        Y_validation = dataset.make_ydata(idx_validation).ravel()
        
        ## 2-2) Create a model
        clf = RandomForestClassifier(n_estimators=N_ESTIMTORS,
                                     max_depth=MAX_DEPTH,
                                     n_jobs=None)
                    
        ## 2-3) Fit a model
        history = clf.fit(X_construction, Y_construction)
        
        ## 2-4) Compute the metric
        Pred_validation = clf.predict(X_validation)
        objective_metrics += accuracy_score(Y_validation, Pred_validation)
    
    training_accuracy = objective_metrics / kf.get_n_splits()
    
    if verbose > 1:
        print('[INNER] [{}/{}] N_ESTIMTORS: {:d}'.format(fitness_step, innerkfold, N_ESTIMTORS))
        print('[INNER] [{}/{}] MAX_DEPTH: {:d}'.format(fitness_step, innerkfold, MAX_DEPTH))
        print('[INNER] [{}/{}] TRAINING_ACCURACY: {:.3f}'.format(fitness_step, innerkfold, training_accuracy))
    
    fitness_step += 1
    return -training_accuracy


    
    
if __name__=="__main__":
    main()
    