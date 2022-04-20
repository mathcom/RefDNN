import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform


class DATASET:
    def __init__(self, drfile, gefile, fpfile):
        ## 1. Read data
        ## 1-1) Gene expression
        self.ge = pd.read_csv(gefile, sep=',', index_col=0)
        ## 1-2) Drug response
        self.dr = pd.read_csv(drfile, sep=',', dtype='str')
        self.DRUGKEY = self.dr.columns[0]
        self.CELLKEY = self.dr.columns[1]
        self.LABELKEY = self.dr.columns[2]
        ## 1-3) Fingerprint
        self.fp = pd.read_csv(fpfile, sep=',', index_col=0).transpose()
        
        ## 2. Preprocessing
        ## 2-1) Find targets
        target_drugs = self._find_target_drugs()
        target_cells = self._find_target_cells()
        ## 2-2) Filter data
        self.ge = self.ge.filter(target_cells)
        self.dr = self.dr[self._get_target_idx(target_drugs, target_cells)]
        ## 2-3) Label string to integer
        idx = self.dr[self.LABELKEY] == 'Resistance'
        self.dr[self.LABELKEY][idx] = 1
        self.dr[self.LABELKEY][~idx] = 0
        self.dr[self.LABELKEY] = self.dr[self.LABELKEY].astype(np.uint8)
        
        ## 3. Structural Similarity Profile
        self.SSP = self._make_SSP()

    def __len__(self):
        return len(self.dr)
        
    def get_drugs(self, unique=False):
        return self._get_series(self.DRUGKEY, unique)
        
    def get_cells(self, unique=False):
        return self._get_series(self.CELLKEY, unique)
        
    def get_labels(self, unique=False):
        return self._get_series(self.LABELKEY, unique)
        
    def get_genes(self):
        return self.ge.index.values
        
    def get_exprs(self):
        return self.ge.values.T # per cell
    
    def make_xdata(self, idx=None):
        if idx is None:
            cells = self.get_cells()
        else:
            cells = self.get_cells()[idx]
        return np.array([self.ge[cell].values for cell in cells], dtype=np.float32)
    
    def make_ydata(self, idx=None):
        if idx is None:
            labels = self.get_labels()
        else:
            labels = self.get_labels()[idx]
        return np.expand_dims(np.array(labels, dtype=np.uint8), axis=-1)
    
    def make_sdata(self, base_drugs, idx=None):
        SSP = self.SSP.filter(base_drugs, axis='index')
        if idx is None:
            drugs = self.get_drugs()
        else:
            drugs = self.get_drugs()[idx]
        return np.array([SSP[drug].values for drug in drugs], dtype=np.float32)
        
    def make_idata(self, base_drugs, idx=None):
        drug2idx = {drug:i for i, drug in enumerate(base_drugs)}
        if idx is None:
            drugs = self.get_drugs()
        else:
            drugs = self.get_drugs()[idx]
        return np.array([drug2idx[drug] for drug in drugs], dtype=np.int64)
    
    def _make_SSP(self):
        ssp_mat = 1. - squareform(pdist(self.fp.values.T, 'jaccard'))
        return pd.DataFrame(ssp_mat, index=self.fp.columns, columns=self.fp.columns)
        
    def _get_series(self, KEY, unique):
        if unique:
            return np.sort(self.dr[KEY].unique(), kind='mergesort')
        else:
            return self.dr[KEY].values
    
    def _find_target_drugs(self):
        target_drugs = []
        self.drug2fp = {}
        for drugname in self.dr[self.DRUGKEY].unique():
            if drugname in self.fp:
                target_drugs.append(drugname)
                self.drug2fp[drugname] = self.fp[drugname].astype(np.uint8).astype(bool)
        return target_drugs
        
    def _find_target_cells(self):
        return self.ge.columns.intersection(self.dr[self.CELLKEY].unique())

    def _get_target_idx(self, target_drugs, target_cells):
        idx_drugs = self.dr[self.DRUGKEY].isin(target_drugs)
        idx_cells = self.dr[self.CELLKEY].isin(target_cells)
        return idx_drugs & idx_cells    
    
    
def read_cancertypeFile(filepath):
    with open(filepath) as fin:
        lines = fin.readlines()
    lines = [line.rstrip().split('\t') for line in lines]
    '''
    0 Sample Name
    1 COSMIC identifier
    2 Whole Exome Sequencing (WES)
    3 Copy Number Alterations (CNA)
    4 Gene Expression
    5 Methylation
    6 Drug Response
    7 GDSC Tissue descriptor 1
    8 GDSC Tissue descriptor 2
    9 Cancer Type (matching TCGA label)
    10 Microsatellite instability Status (MSI)
    11 Screen Medium
    12 Growth Properties
    '''
    res = {}
    for line in lines[1:-1]:
        cell = line[1]
        cancertype = line[9]
        if cancertype in {'', 'UNABLE TO CLASSIFY'}:
            cancertype = 'others'
        res[cell] = cancertype
        
    print(set(res.values()))
    
    return res