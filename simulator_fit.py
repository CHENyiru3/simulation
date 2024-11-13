# import library
import numpy as np
import pandas as pd


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

def nb_loglikelihood_fixed_mu(r, x, mu):
    p = r / (r + mu)
    return -np.sum(stats.nbinom.logpmf(x, r, p))

def zinb_loglikelihood_fixed_mu(params, x, mu):
    pi, r = params
    p = r / (r + mu)
    x_is_zero = (x == 0)
    ll_zero = x_is_zero * np.log(pi + (1 - pi) * stats.nbinom.pmf(0, r, p))
    ll_nonzero = ~x_is_zero * (np.log(1 - pi) + stats.nbinom.logpmf(x, r, p))
    return -np.sum(ll_zero + ll_nonzero)

def zip_loglikelihood_fixed_mu(pi, x, mu):
    x_is_zero = (x == 0)
    ll_zero = x_is_zero * np.log(pi + (1 - pi) * np.exp(-mu))
    ll_nonzero = ~x_is_zero * (np.log(1 - pi) + stats.poisson.logpmf(x, mu))
    return -np.sum(ll_zero + ll_nonzero)

def poisson_loglikelihood(mu, x):
    return -np.sum(stats.poisson.logpmf(x, mu))

def fit_with_simulated_mean_and_var(gene, simulated_mean, simulated_var, maxiter=100):
    mu = simulated_mean
    
    # 计算负二项分布参数
    if simulated_var < simulated_mean:  # 如果方差小于均值，使用泊松分布
        return [0, np.inf, mu, "Poisson"]
    else:
        p = (simulated_var - mu) / simulated_var
        r = mu * (1 - p) / p
        
        # 计算各个模型的似然值
        ll_nb = -nb_loglikelihood_fixed_mu(r, gene, mu)
        
        result_zinb = minimize(zinb_loglikelihood_fixed_mu, [0.5, r], 
                             args=(gene, mu), 
                             bounds=[(1e-6, 1-1e-6), (1e-6, 1e6)])
        pi_zinb, r_zinb = result_zinb.x
        ll_zinb = -result_zinb.fun

        result_zip = minimize_scalar(zip_loglikelihood_fixed_mu, 
                                   args=(gene, mu), 
                                   bounds=(1e-6, 1-1e-6), 
                                   method='bounded')
        pi_zip = result_zip.x
        ll_zip = -result_zip.fun

        ll_poisson = -poisson_loglikelihood(mu, gene)

        # 计算AIC
        aic_nb = 2 * 2 - 2 * ll_nb
        aic_zinb = 2 * 3 - 2 * ll_zinb
        aic_zip = 2 * 2 - 2 * ll_zip
        aic_poisson = 2 * 1 - 2 * ll_poisson
        
        aics = [aic_nb, aic_zinb, aic_zip, aic_poisson]
        best_model_idx = np.argmin(aics)

        if best_model_idx == 0:
            return [0, r, mu, "NB"]
        elif best_model_idx == 1:
            return [pi_zinb, r_zinb, mu, "ZINB"]
        elif best_model_idx == 2:
            return [pi_zip, np.inf, mu, "ZIP"]
        else:
            return [0, np.inf, mu, "Poisson"]

def fit_marginal_model_with_simulated_params(adata, simulated_means, simulated_vars, 
                                           min_nonzero_num=2, maxiter=500, n_jobs=-1):
    if not isinstance(adata, anndata.AnnData):
        raise ValueError("Input adata should be an AnnData object")
    
    if sp.issparse(adata.X):
        x = adata.X.toarray()
    else:
        x = adata.X
    
    gene_names = adata.var_names.tolist()
    n, p = x.shape
    
    if len(simulated_means) != p or len(simulated_vars) != p:
        raise ValueError("Length of simulated parameters does not match number of genes")
    
    gene_zero_prop = 1 - np.sum(x > 0, axis=0) / n
    genes = np.where(gene_zero_prop < 1 - min_nonzero_num / n)[0]
    
    if len(genes) == 0:
        print("Warning: No genes selected for fitting models.")
        return None
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(fit_with_simulated_mean_and_var)(
            x[:, i], 
            simulated_means[gene_names[i]], 
            simulated_vars[gene_names[i]], 
            maxiter
        )
        for i in tqdm(genes, desc="Fitting models")
    )
    
    params_df = pd.DataFrame(results, 
                           index=[gene_names[i] for i in genes], 
                           columns=['pi0', 'theta', 'mu', 'model_selected'])
    
    model_params = {
        'genes': {i: gene_names[i] for i in genes},
        'marginal_param1': params_df[['pi0', 'theta', 'mu']].values.tolist(),
        'model_selected': params_df['model_selected'].tolist(),
        'min_nonzero_num': min_nonzero_num,
        'n_cell': n,
        'n_read': np.sum(x)
    }
    
    return model_params