# UFGOT
Alignment of two data can be achieved using the ufgot function in solver.py,where the input $X∈dx×n$ is the matrix that needs to be aligned, $Y∈dy×n$ is the target alignment matrix, and the fiter parameter allows you to select from g1, g2... g6 selects the filter operator. If the split parameter is True, the entire data set will be randomly divided into three parts to train the optimal alignment. Otherwise, the entire data set will be used for training. The p1 and p2 parameters allow you to customize the input hyperparameter range. This function will return the optimal alignment result $X_{alig}∈dy×n$ and the target alignment matrix $Y$ within the range of the hyperparameters. The alignment results can be obtained using the foscttm function.
```
X_alig, Y = x_alig,y = ufgot(X, Y, fiter='g1', split=True)
foscttm(x_alig, y)
```
Use the UFGOT_cancer function in solver.py to implement aligned ensemble cluster analysis of cancer subtype data. The parameter cancer is used to select the cancer data set for the experiment. If the data set provides real labels, the parameter k is the number of categories. This function returns aligned CNV, Methy, miRNA data and target alignment data mRNA as well as 11 combined clustering results. The code provides a set of alignment parameters for the trained COAD data set, which can be used directly without training.
```
CNV, Methy, miRNA, mRNA, res = UFGOT_cancer(cancer='COAD', k=4)
```
res is the clustering result
