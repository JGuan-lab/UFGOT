from slover import ufgot, UFGOT_cancer
from eval import foscttm

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

CNV, Methy, miRNA, mRNA, res = UFGOT_cancer(cancer='COAD', k=4)
print(foscttm(CNV, mRNA))
print(foscttm(Methy, mRNA))
print(foscttm(miRNA, mRNA))
print(res)
