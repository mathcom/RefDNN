RefDNN: a reference drug based neural network for more accurate prediction of anticancer drug resistance

We introduce a Reference Drug-based Neural Detwork model (RefDNN) for predicting anticancer drug resistance and identifying biomarkers contributing drug resistance.

RefDNN can predict accurately whether a cell line is resistant to a new drug on which the model was not trained.

RefDNN requires gene expression profiles and drug molecular fingerprint data.

For more detail, please refer to Choi, Jonghwan, et al. "RefDNN: a reference drug based neural network for more accurate prediction of anticancer drug resistance." (submitted)


* Latest update: 6 April 2019

--------------------------------------------------------------------------------------------
SYSTEM REQUIERMENTS: 

    RefDNN requires system memory larger than 24GB.
    
    If you want to use tensorflow-gpu, GPU memory of more than 4GB is required.


--------------------------------------------------------------------------------------------
PYTHON LIBRARY REQUIERMENTS:

    If you want to use only tensorflow-cpu:

        pip install -r requirements_cpu.txt
        
    If you want to use tensorflow-gpu:
    
        pip install -r requirements_gpu.txt
    
--------------------------------------------------------------------------------------------
USAGE: 

    If you want to use only tensorflow-cpu:

        $ python 1_nested_cv_baysian_search.py data/response_GDSC.csv data/expression_GDSC.csv data/fingerprint_GDSC.csv
        
    If you want to use tensorflow-gpu:
    
        $ python 1_nested_cv_baysian_search.py data/response_GDSC.csv data/expression_GDSC.csv data/fingerprint_GDSC.csv -g
    

--------------------------------------------------------------------------------------------
NOTE:

    The option parameter '-h' shows help message.
    
    $ python 1_nested_cv_baysian_search.py -h
    
    
--------------------------------------------------------------------------------------------
