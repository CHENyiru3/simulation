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


def simulate_gene(iter, gene_names, adata, model_params, rr):
    gene_name = gene_names[iter]
    if gene_name in adata.var_names:
        gene_expr = adata[:, gene_name].X.toarray().flatten()

        print(f"\nSimulating gene: {gene_name}")
        print(f"Original stats - Min: {gene_expr.min():.2f}, Max: {gene_expr.max():.2f}, Mean: {gene_expr.mean():.2f}")

        original_order = np.argsort(gene_expr)
        param = model_params['marginal_param1'][iter]
        model_type = model_params['model_selected'][iter]

        param = [float(p) if p != 'inf' else np.inf for p in param]

        try:
            if model_type == 'Poisson':
                lambda_param = param[2] * rr
                sim_raw_expr = poisson.rvs(lambda_param, size=adata.shape[0])
            elif model_type == 'NB':
                r_param = param[1]
                if np.isinf(r_param):
                    lambda_param = param[2] * rr
                    sim_raw_expr = poisson.rvs(lambda_param, size=adata.shape[0])
                else:
                    p_param = r_param / (r_param + param[2] * rr)
                    r_param = np.maximum(r_param, 1e-8)
                    p_param = np.clip(p_param, 1e-8, 1 - 1e-8)
                    sim_raw_expr = nbinom.rvs(r_param, p_param, size=adata.shape[0])
            elif model_type == 'ZIP':
                pi0 = param[0]
                lambda_param = param[2] * rr
                zero_mask = bernoulli.rvs(pi0, size=adata.shape[0])
                sim_raw_expr = poisson.rvs(lambda_param, size=adata.shape[0]) * (1 - zero_mask)
            elif model_type == 'ZINB':
                pi0 = param[0]
                r_param = param[1]
                p_param = r_param / (r_param + param[2] * rr)

                if r_param <= 0 or not (0 < p_param < 1):
                    raise ValueError(f"Invalid parameters for nbinom: r = {r_param}, p = {p_param}")
                
                zero_mask = bernoulli.rvs(pi0, size=adata.shape[0])
                sim_raw_expr = nbinom.rvs(r_param, p_param, size=adata.shape[0]) * (1 - zero_mask)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
        except Exception as e:
            print(f"Warning: Error simulating gene {gene_name} with {model_type} model: {e}")
            print("Falling back to Poisson distribution with mean as lambda")
            # 使用原始数据的平均值作为 Poisson 分布的参数
            mean_expr = np.mean(gene_expr)
            sim_raw_expr = poisson.rvs(mean_expr, size=adata.shape[0])
        
        try:
            sim_order = np.argsort(sim_raw_expr)
            final_expr = np.zeros_like(gene_expr)
            final_expr[original_order] = sim_raw_expr[sim_order]

            top_10_percent = np.percentile(gene_expr, 90)
            original_high = gene_expr > top_10_percent
            final_high = final_expr > top_10_percent
            overlap = np.sum(original_high & final_high) / np.sum(original_high)
            print(f"Top 10% preservation: {overlap * 100:.2f}%")

            correlation = np.corrcoef(gene_expr, final_expr)[0, 1]
            print(f"Correlation: {correlation:.4f}")

            return final_expr
        except Exception as e:
            print(f"Error in post-processing for gene {gene_name}: {e}")
            # 如果后处理也失败，直接返回 Poisson 模拟结果
            return sim_raw_expr
    else:
        return np.zeros(adata.shape[0])

def simulator_count_single(model_params, adata, rr=1, breaktie='random', num_cores=1):
    p = len(model_params['genes'])
    gene_names = [None] * p

    for idx, (i, gene_name) in enumerate(model_params['genes'].items()):
        if idx < p:
            gene_names[idx] = gene_name
        else:
            raise IndexError(f"Index {idx} out of range for genes")

    num_loc, num_genes = adata.shape
    result = np.zeros((p, num_loc), dtype=float)

    if p > 0:
        for iter in range(p):
            result[iter, :] = simulate_gene(iter, gene_names, adata, model_params, rr)

    return result

def simulator_remain_simulate_count(simulator, adata, breaktie='random', rrr=None, num_cores=8, verbose=False):
    if simulator.simcolData is None:
        simulator.simcolData = simulator.refcolData.copy()

    oldnum_loc = simulator.refcolData.shape[0]
    newnum_loc = simulator.simcolData.shape[0]
    param_res = simulator.EstParam[0] 

    # Calculate total counts in the old data
    total_count_old = adata.obs['total_counts'].sum()
    total_count_new = total_count_old  
    r = (total_count_new / newnum_loc) / (total_count_old / oldnum_loc) if rrr is None else rrr

    if verbose:
        print(f"The ratio between the seqdepth per location is: {r}")

    # Generate the simulated count matrix
    rawcount = simulator_count_single(model_params=param_res, adata=adata, rr=r, breaktie=breaktie, num_cores=num_cores)
    
    # Handle all-zero genes
    all_zero_idx = np.where(np.sum(rawcount, axis=1) == 0)[0]
    if len(all_zero_idx) > 0:
        for idx in all_zero_idx:
            nonzero_idx = np.random.choice(newnum_loc, 1)[0]
            rawcount[idx, nonzero_idx] = 1

    outcount = np.round(rawcount).astype(int)
    simulator.simCounts = sp.csr_matrix(outcount)

    return simulator