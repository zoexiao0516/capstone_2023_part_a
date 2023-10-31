import pandas as pd
from low_rank_capacity import *

DATAFRAMES_PATH = '/mnt/home/cchou/ceph/Capstone/Dataframes/'
epoch = 42
df = pd.read_csv(DATAFRAMES_PATH+str(epoch))

print(df['X_projected'].apply(manifold_analysis_corr))