# UFGOT
Alignment of two data can be achieved using the ufgot function in solver.py, where the input $X∈dx×n$ is the matrix that needs to be aligned, $Y∈dy×n$ is the target referenced matrix, and the filter parameter allows you to select the filter operator from g1, g2, ... , g6. If the split parameter is True, the entire data set will be randomly divided into three parts to train the optimal alignment. Otherwise, the entire data set will be used for training. The p1 and p2 parameters allow you to customize the input hyperparameter range. This ufgot function will return the optimal alignment result $X_{alig}∈dy×n$ and the target referenced matrix $Y$ within the range of the hyperparameters. The alignment performance results can be obtained using the foscttm function.
```
from slover import ufgot
from eval import foscttm
X_alig, Y = ufgot(X, Y, filter='g1', split=True)
foscttm(X_alig, Y)
```
Use the UFGOT_cancer function in solver.py to implement cluster analysis of cancer subtype data. The parameter cancer is used to select the cancer data set for the experiment. If the data set provides real labels, the parameter k is the number of categories. This function returns aligned CNV, Methy, miRNA data and target referenced data mRNA as well as the clustering results of 11 combinations of omics-data. The code provides a set of alignment parameters for the trained COAD data set, which can be used directly without training.
```
from slover import UFGOT_cancer
from eval import foscttm
CNV, Methy, miRNA, mRNA, res = UFGOT_cancer(cancer='COAD', k=4)
foscttm(CNV, mRNA)
foscttm(Methy, mRNA)
foscttm(miRNA, mRNA)
```
res is the clustering result
