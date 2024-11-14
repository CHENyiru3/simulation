# import library
import numpy as np
import pandas as pd
import anndata

from scipy import sparse as sp
from scipy import stats
from scipy.stats import (
    genpareto, 
    ks_2samp, 
    nbinom, 
    poisson, 
    bernoulli, 
    invgamma
)
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial.distance import cdist
from scipy.special import gammaln


import anndata as ad
import scanpy as sc


from multiprocessing import Pool
from joblib import Parallel, delayed


from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

# 忽略警告
import warnings
warnings.filterwarnings('ignore')
import pickle

# import from other modules
from var_mean_sim import simulate_gene_average_expression, simulate_gene_variances_advanced
from simulator_generation import simulator_remain_simulate_count
from simulator_fit import fit_marginal_model_with_simulated_params


class SimulatorSRT:
    def __init__(self, adata, model_params):
        self.refCounts = adata.to_df()  
        self.refcolData = adata.obs.copy()  
        self.simcolData = None
        self.EstParam = [model_params]
        self.simCounts = None
def run_simulation_tissue(adata):
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    sc.pp.calculate_qc_metrics(adata, inplace=True)

    all_genes = adata.var_names.tolist()
    print(adata)

    simulated_means = simulate_gene_average_expression(adata)
    # 模拟方差
    simulated_vars, var_threshold, var_evaluation= simulate_gene_variances_advanced(adata)
    
    print(f"Variance simulation - Best threshold: {var_threshold}")
    print(f"Variance simulation - Evaluation: {var_evaluation}")

    print(f"Number of simulated genes: {len(simulated_means)}")
    # 使用模拟的参数拟合边际模型
    model_params = fit_marginal_model_with_simulated_params(
        adata, 
        simulated_means, 
        simulated_vars, 
        min_nonzero_num=2, 
        maxiter=500, 
        n_jobs=-1
    )

    model_params['simulation_evaluation'] = {
        'variance': var_evaluation,

    }

    with open('/Users/chen_yiru/Desktop/simulation/data/151676_model.pkl', 'wb') as f:
        pickle.dump(model_params, f)
    
    return model_params

    return model_params
def simulation_slice(adata):
    model_params = run_simulation_tissue(adata)
    simulator = SimulatorSRT(adata, model_params)

    simulated_simulator = simulator_remain_simulate_count(simulator, adata, num_cores=8, verbose=True)

    simulated_counts = simulated_simulator.simCounts

    if simulated_counts.shape != adata.shape:
        print(f"Warning: simulated_counts shape {simulated_counts.shape} does not match adata shape {adata.shape}")
        if simulated_counts.shape == (adata.shape[1], adata.shape[0]):
            simulated_counts = simulated_counts.T  
        elif simulated_counts.shape != adata.shape:
            raise ValueError("Cannot adjust simulated_counts shape to match adata shape")

    simulated_adata = anndata.AnnData(
        X=simulated_counts,
        obs=adata.obs.copy(),
        var=adata.var.copy(),
        obsm={'spatial': adata.obsm['spatial']}
    )

    simulated_adata.obs['total_counts'] = simulated_adata.X.sum(axis=1)
    simulated_adata.obs['n_genes'] = (simulated_adata.X > 0).sum(axis=1)

    return simulated_adata


# test
adata = sc.read_h5ad('/Users/chen_yiru/Desktop/simulation/data/raw/Sample_data_151676.h5ad')
print(adata)
simulated_adata = simulation_slice(adata)
print(simulated_adata)