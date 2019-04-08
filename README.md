#RefDNN: a reference drug based neural network for more accurate prediction of anticancer drug resistance

We introduce a Reference Drug-based Neural Detwork model (RefDNN) for predicting anticancer drug resistance and identifying biomarkers contributing drug resistance.

RefDNN can predict accurately whether a cell line is resistant to a new drug on which the model was not trained.

RefDNN requires gene expression profiles and drug molecular fingerprint data.

For more detail, please refer to Choi, Jonghwan, et al. "RefDNN: a reference drug based neural network for more accurate prediction of anticancer drug resistance." (submitted)


* Latest update: 8 April 2019

--------------------------------------------------------------------------------------------
##SYSTEM REQUIERMENTS: 

    - RefDNN requires system memory larger than 24GB.
    
    - If you want to use tensorflow-gpu, GPU memory of more than 4GB is required.


--------------------------------------------------------------------------------------------
##PYTHON LIBRARY REQUIERMENTS:

    - If you want to use only tensorflow-cpu:

        $ pip install -r requirements_cpu.txt
        
    - If you want to use tensorflow-gpu:
    
        $ pip install -r requirements_gpu.txt
    
--------------------------------------------------------------------------------------------
##USAGE: 

    - If you want to use only tensorflow-cpu:

        $ python 1_nested_cv_baysian_search.py data/response_GDSC.csv data/expression_GDSC.csv data/fingerprint_GDSC.csv
        
    - If you want to use tensorflow-gpu:
    
        $ python 1_nested_cv_baysian_search.py data/response_GDSC.csv data/expression_GDSC.csv data/fingerprint_GDSC.csv -g
    
--------------------------------------------------------------------------------------------
##EXAMPLE:

    - Note that the complete time of this example is about 8 hours.
    
    - When running the source code of nested cross validation with CCLE dataset, an user can see the following logs.
    
        $ python 1_nested_cv_baysian_search.py data/response_CCLE.csv data/expression_CCLE.csv data/fingerprint_CCLE.csv -g -o output_1_CCLE
        [DATA INFO] num_pairs: 5724
        [DATA INFO] num_drugs: 12
        [DATA INFO] num_cells: 491
        [DATA INFO] num_genes: 18926
        [DATA INFO] num_sensitivity: 2322
        [DATA INFO] num_resistance: 3402
        [OUTER][1/5] NOW TUNING THE MODEL USING BAYESIAN OPTIMIZATION...
        [INNER] NOW EVALUATING PARAMETERS IN THE INNER LOOP...
        [INNER][01/20] hidden_units: 18
        [INNER][01/20] learning_rate_ftrl: 5.416e-02
        [INNER][01/20] learning_rate_adam: 1.077e-05
        [INNER][01/20] l1_regularization_strength: 2.704e+01
        [INNER][01/20] l2_regularization_strength: 2.341e-01
        [INNER][01/20] training_accuracy: 0.806
        [INNER] NOW EVALUATING PARAMETERS IN THE INNER LOOP...
        [INNER][02/20] hidden_units: 67
        [INNER][02/20] learning_rate_ftrl: 4.289e-02
        [INNER][02/20] learning_rate_adam: 4.991e-04
        [INNER][02/20] l1_regularization_strength: 5.452e-01
        [INNER][02/20] l2_regularization_strength: 3.942e-03
        [INNER][02/20] training_accuracy: 0.618
        [INNER] NOW EVALUATING PARAMETERS IN THE INNER LOOP...
        [INNER][03/20] hidden_units: 28
        [INNER][03/20] learning_rate_ftrl: 4.909e-03
        
        (...intermediate omission...)
        
        [INNER][19/20] learning_rate_adam: 6.794e-06
        [INNER][19/20] l1_regularization_strength: 1.000e+02
        [INNER][19/20] l2_regularization_strength: 1.000e-03
        [INNER][19/20] training_accuracy: 0.899
        [INNER] NOW EVALUATING PARAMETERS IN THE INNER LOOP...
        [INNER][20/20] hidden_units: 81
        [INNER][20/20] learning_rate_ftrl: 3.783e-06
        [INNER][20/20] learning_rate_adam: 4.434e-06
        [INNER][20/20] l1_regularization_strength: 1.000e-03
        [INNER][20/20] l2_regularization_strength: 1.000e-03
        [INNER][20/20] training_accuracy: 0.885
        [OUTER][5/5] BEST_HIDDEN_UNITS : 4
        [OUTER][5/5] BEST_LEARNING_RATE_FTRL : 2.540e-03
        [OUTER][5/5] BEST_LEARNING_RATE_ADAM : 1.000e-01
        [OUTER][5/5] BEST_L1_REGULARIZATION_STRENGTH : 1.000e+02
        [OUTER][5/5] BEST_L2_REGULARIZATION_STRENGTH : 1.000e+02
        [OUTER][5/5] BEST_TRAINING_ACCURACY : -0.912
        [OUTER][5/5] NOW TRAINING THE MODEL WITH BEST PARAMETERS...
        [OUTER][5/5] BEST_TEST_ACCURACY : 0.918
        [OUTER][5/5] BEST_TEST_AUCROC : 0.964
        [OUTER][5/5] BEST_TEST_AUCPR : 0.970
        FINISH

    - This example will make a directory termed 'output_1_CCLE' and save a result file named 'metrics_hyperparameters.csv' in the directory.
    
    - The result file contains 3 metrics and 5 hyperparameter values computed by 5-fold cross validation.

--------------------------------------------------------------------------------------------
##NOTE:

    The option parameter '-h' shows help message.
    
        $ python 1_nested_cv_baysian_search.py -h
    
    
--------------------------------------------------------------------------------------------
