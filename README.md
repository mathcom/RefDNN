# RefDNN: a reference drug based neural network for more accurate prediction of anticancer drug resistance

We introduce a Reference Drug-based Neural Detwork model (RefDNN) for predicting anticancer drug resistance and identifying biomarkers contributing drug resistance.

RefDNN can predict accurately whether a cell line is resistant to a new drug on which the model was not trained.

RefDNN requires gene expression profiles and drug molecular fingerprint data.

For more detail, please refer to Choi, Jonghwan, et al. "RefDNN: a reference drug based neural network for more accurate prediction of anticancer drug resistance." Scientific Reports 10 (2020):1-11


* Latest update: 06 February 2020

--------------------------------------------------------------------------------------------
## SYSTEM REQUIERMENTS: 

    - RefDNN requires system memory larger than 24GB.
    
    - If you want to use tensorflow-gpu, GPU memory of more than 4GB is required.


--------------------------------------------------------------------------------------------
## PYTHON LIBRARY REQUIERMENTS:

    - If you want to use only tensorflow-cpu:

        $ pip install -r requirements_cpu.txt
        
    - If you want to use tensorflow-gpu:
    
        $ pip install -r requirements_gpu.txt
    

--------------------------------------------------------------------------------------------
## HELP MESSAGES:

    - The option parameter '-h' shows help message.
    
        $ python 1_nested_cv_baysian_search.py -h
    
        usage: 1_nested_cv_baysian_search.py [-h] [-o outputdir] [-b batchsize]
                                             [-t numtrainingsteps]
                                             [-s numbayesiansearch] [-k outerkfold]
                                             [-l innerkfold] [-v verbose]
                                             responseFile expressionFile
                                             fingerprintFile

        positional arguments:
          responseFile          A filepath of drug response data for TRAINING
          expressionFile        A filepath of gene expression data for TRAINING
          fingerprintFile       A filepath of fingerprint data for TRAINING

        optional arguments:
          -h, --help            show this help message and exit
          -o outputdir          A directory path for saving outputs
                                (default:'output_1')
          -b batchsize          A size of batch on training process. The small size is
                                recommended if an available size of RAM is small
                                (default: 64)
          -t numtrainingsteps   Number of training steps on training process. It is
                                recommended that the steps is larger than (numpairs /
                                batchsize) (default: 5000)
          -s numbayesiansearch  Number of bayesian search for hyperparameter tuning
                                (default: 20)
          -k outerkfold         K for outer k-fold cross validation (default: 5)
          -l innerkfold         L for inner l-fold cross validation (default: 3)
          -v verbose            0:No logging, 1:Basic logging to check process, 2:Full
                                logging for debugging (default:1)

    
--------------------------------------------------------------------------------------------
## EXAMPLE:

    - This example shows how to run the nested cross validation on the CCLE dataset.
    
    - Since the default argument values of source code produce very very long process time(about 24 hours), we modify some values.
    
        $ python 1_nested_cv_baysian_search.py data/response_CCLE.csv data/expression_CCLE.csv data/fingerprint_CCLE.csv -o output_1_CCLE -s 10 -t 1000 -b 32

    - The example requires about 30 minutes if GPU is exploited.
    
    - This example will make a directory named 'output_1_CCLE' and save a result file named 'metrics_hyperparameters.csv' in the directory.
    
    - The result file contains 3 metrics and hyperparameter values computed by the nested cross validation.
    
    - When running the example, logging information like below is found.
    
        [START]
        [DATA] NUM_PAIRS: 5724
        [DATA] NUM_DRUGS: 12
        [DATA] NUM_CELLS: 491
        [DATA] NUM_GENES: 18926
        [DATA] NUM_SENSITIVITY: 2322
        [DATA] NUM_RESISTANCE: 3402
        [TIME] [1] 2019-4-12 15:16:18
        [TIME] [2] 2019-4-12 15:16:18
        [OUTER] [1/5] NOW TUNING THE MODEL USING BAYESIAN OPTIMIZATION...
        [OUTER] [1/5] BEST_HIDDEN_UNITS : 6
        [OUTER] [1/5] BEST_LEARNING_RATE_FTRL : 1.000e-01
        [OUTER] [1/5] BEST_LEARNING_RATE_ADAM : 6.827e-02
        [OUTER] [1/5] BEST_L1_REGULARIZATION_STRENGTH : 7.324e+00
        [OUTER] [1/5] BEST_L2_REGULARIZATION_STRENGTH : 1.052e-01
        [OUTER] [1/5] BEST_TRAINING_ACCURACY : -0.908
        [OUTER] [1/5] NOW TRAINING THE MODEL WITH BEST PARAMETERS...
        [OUTER] [1/5] BEST_TEST_ACCURACY : 0.907
        [OUTER] [1/5] BEST_TEST_AUCROC : 0.926
        [OUTER] [1/5] BEST_TEST_AUCPR : 0.940
        [TIME] [3] 2019-4-12 15:22:11
        [OUTER] [2/5] NOW TUNING THE MODEL USING BAYESIAN OPTIMIZATION...
        
        (...intermediate omission...)
        
        [OUTER] [5/5] NOW TRAINING THE MODEL WITH BEST PARAMETERS...
        [OUTER] [5/5] BEST_TEST_ACCURACY : 0.898
        [OUTER] [5/5] BEST_TEST_AUCROC : 0.944
        [OUTER] [5/5] BEST_TEST_AUCPR : 0.951
        [TIME] [3] 2019-4-12 15:45:36
        [TIME] [4] 2019-4-12 15:45:36
        [FINISH]


--------------------------------------------------------------------------------------------
