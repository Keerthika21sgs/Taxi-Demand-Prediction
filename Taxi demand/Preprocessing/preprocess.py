import numpy as np
#from tqdm import tqdm
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt


taxidf = pd.read_csv('./dataset/taxi_data.csv', index_col=0)
taxidf.index = pd.DatetimeIndex(taxidf.index)

print(taxidf.index.min())
print(taxidf.index.max())

dat_index = pd.date_range(taxidf.index.min(), taxidf.index.max(), freq="1H")
taxidf = taxidf.reindex(dat_index)
taxidf["miss_date"] = np.NaN
print("T1 " , taxidf.loc[taxidf.isnull().all(1)])
taxidf.loc[taxidf.isnull().all(1) == True, "miss_date"] = True
taxidf.loc[~(taxidf.miss_date == True), "miss_date"] = False
print(taxidf.loc[taxidf["miss_date"] == True, :].shape)
print(taxidf.loc[taxidf["miss_date"] == False, :].shape)
print(taxidf.loc[taxidf["miss_date"] == True, :].shape)
print("T1 " , taxidf.loc[taxidf["miss_date"] == True, :])

miss_hrs = taxidf[(taxidf.miss_date == True)].index.hour
pd.Series(Counter(miss_hrs)).plot(kind="bar")
plt.show()
index_missing_ = taxidf.loc[(taxidf.miss_date == True) & 
                                                    (taxidf.index.hour.isin((0, 1, 2, 3, 4, 5, 6))), :
                                                   ].index
taxidf.loc[index_missing_, "num_pickups"] = 0
taxidf.loc[index_missing_, "num_passengers"] = 0
taxidf.loc[index_missing_, "miss_date"] = False
index_missing_day = taxidf.loc[(taxidf.miss_date == True) & 
                                                    (taxidf.index.hour > 6), :].index
hour_mean = taxidf.groupby([taxidf.index.month, taxidf.index.dayofweek, taxidf.index.hour]).mean()
idxslice = pd.IndexSlice
for idx in index_missing_day:
    taxidf.loc[idx, "num_pickups"] = hour_mean.loc[idxslice[idx.month, idx.dayofweek, idx.hour], 
                                                                    "num_pickups"]
    taxidf.loc[idx, "num_passengers"] = hour_mean.loc[idxslice[idx.month, idx.dayofweek, idx.hour], 
                                                                    "num_passengers"]
    taxidf.loc[idx, "miss_date"] = False
print(taxidf.loc[index_missing_day, :])
