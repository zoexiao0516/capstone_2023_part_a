import argparse
import sys
import os
sys.path.append('../../')

import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import capacity.geometry_correlations as new
import capacity.mean_field_cap as PRX
import capacity.basic as basic
from tqdm import tqdm
from mpi4py import MPI
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from capacity.utils import *

rng = np.random.default_rng()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nthreads = comm.Get_size()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch', default='1')
    
    args = parser.parse_args()
    
    epoch = int(args.epoch)
    print(f'Epoch: {epoch}')
    
    # Load data
    path = f'/mnt/home/cchou/ceph/Capstone/Dataframes/epoch{epoch}.pkl'
    data = pd.read_pickle(path)
    
    M_small = 50
    num_rep = 1
    
    idx = pd.IndexSlice
    
    df = pd.DataFrame()
    result_list = []

    # Run the analysis
    for layer in range(5):
        XtotT = data.loc[idx[layer,'X_projected']]
        XtotT = zscore_XtotT(XtotT)
        P = len(XtotT)
        N, M = XtotT[0].shape

        for i_rep in range(num_rep):

            XtotT_small = [XtotT[i][:,rng.choice([j for j in range(M)],M_small,replace=False)] for i in range(P)]

            result = manifold_analysis_all(XtotT_small)
            result_list.append(result)

            # Save results into a dataframe
            filtered_result = {key: value for key, value in result.items() if not isinstance(value, np.ndarray)}
            tuples = (epoch,layer,i_rep+num_rep*rank)
            index = pd.MultiIndex.from_tuples([tuples], names=['epoch','layer','repeat'])
            df_new = pd.DataFrame(filtered_result, index=index)
            df = pd.concat([df, df_new])

            print(f'Thread {rank} finishes [Layer: {layer}; Repeat: {i_rep}]')

    out_dir = '/mnt/home/cchou/ceph/Capstone/Project_A/result'
    result_id = f'{epoch}_{rank}'
    outfile = os.path.join(out_dir, f'{result_id}.pkl')
    with open(outfile, 'wb') as f:
        pickle.dump(df,f)
        pickle.dump(result_list,f)