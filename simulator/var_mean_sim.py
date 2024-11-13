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


class IG_VarianceSimulator:
    def __init__(self):
        self.alpha = None
        self.beta = None
        self.threshold = None
        self.tail_data = None
        self.original_order = None
        self.original_data = None
        self.random_state = None

    def extract_data(self, X):
        variances = np.var(X, axis=0, ddof=1)
        variances = np.nan_to_num(variances, nan=np.nanmean(variances))
        variances = np.maximum(variances, 1e-10)
        return variances

    def assess_tail_discreteness(self, tail_data):
        sorted_tail = np.sort(tail_data)
        differences = np.diff(sorted_tail)
        cv = np.std(differences) / np.mean(differences)

        if cv > 2.0:
            return 'discrete'
        elif cv < 1.0:
            return 'smooth'
        else:
            return 'mixed'

    def fit_single_component(self, data):
        try:
            params = invgamma.fit(data, floc=0)
            return params[0], params[2]
        except Exception as e:
            print(f"Single component fitting failed: {str(e)}")
            return 1.0, np.mean(data)  # Fallback values

    def fit(self, adata):
        try:
            if sp.issparse(adata.X):
                X = adata.X.toarray()
            else:
                X = adata.X

            self.original_data = self.extract_data(X)
            if not np.all(np.isfinite(self.original_data)):
                raise ValueError("Data contains non-finite values")

            self.original_order = np.argsort(self.original_data)
            sorted_data = np.sort(self.original_data)

            thresholds = [99.5]
            best_score = float('inf')
            best_evaluation = None
            for percentile in thresholds:
                current_threshold = np.percentile(sorted_data, percentile)
                main_data = sorted_data[sorted_data <= current_threshold]
                tail_data = sorted_data[sorted_data > current_threshold]

                if len(main_data) == 0 or len(tail_data) == 0:
                    print(f"Warning: Empty main_data or tail_data at threshold {percentile}")
                    continue

                alpha, beta = self.fit_single_component(main_data)

                n_main = len(main_data)
                n_tail = len(tail_data)

                new_main = invgamma.rvs(alpha, scale=beta, size=n_main, random_state=self.random_state)

                tail_type = self.assess_tail_discreteness(tail_data)
                if tail_type == 'discrete':
                    new_tail = self.random_state.choice(tail_data, size=n_tail, replace=True)
                else:
                    new_tail = np.interp(
                        np.linspace(0, 1, n_tail),
                        np.linspace(0, 1, len(tail_data)),
                        np.sort(tail_data)
                    )

                new_samples = np.concatenate([new_main, new_tail])
                new_samples = np.clip(new_samples, np.min(self.original_data), np.max(self.original_data))

                evaluation = self.evaluate_fit(self.original_data, new_samples)

                score = (abs(evaluation["Cohen's d"]) + 
                         evaluation["Relative Error"] + 
                         evaluation["KS Statistic"] + 
                         (1 - evaluation["Correlation"]))

                if score < best_score:
                    best_score = score
                    self.threshold = percentile
                    self.alpha, self.beta = alpha, beta
                    best_evaluation = evaluation
                    self.tail_data = tail_data

            if self.threshold is None:
                print("Warning: No valid threshold found, fallback to default.")
                self.threshold = 95  # Default threshold

            return self, best_evaluation

        except Exception as e:
            print(f"Fitting failed: {str(e)}")
            self.alpha, self.beta = self.fit_single_component(self.original_data)
            return self, {"Verdict": "Fallback to single component"}

    def simulate(self, n_samples):
        try:
            if self.threshold is None:
                print("Warning: Threshold is None, using default value of 95.")
                self.threshold = 95

            n_main = int(n_samples * self.threshold / 100)
            n_tail = n_samples - n_main

            new_main = invgamma.rvs(self.alpha, scale=self.beta, size=n_main, random_state=self.random_state)

            tail_type = self.assess_tail_discreteness(self.tail_data)
            if tail_type == 'discrete':
                new_tail = self.random_state.choice(self.tail_data, size=n_tail, replace=True)
            else:
                new_tail = np.interp(
                    np.linspace(0, 1, n_tail),
                    np.linspace(0, 1, len(self.tail_data)),
                    np.sort(self.tail_data)
                )

            new_samples = np.concatenate([new_main, new_tail])
            new_samples = np.clip(new_samples, np.min(self.original_data), np.max(self.original_data))
            new_samples = np.sort(new_samples)

            simulated_data = np.zeros_like(new_samples)
            simulated_data[self.original_order] = new_samples

            return simulated_data

        except Exception as e:
            print(f"Simulation failed: {str(e)}")
            return self.random_state.choice(self.original_data, size=n_samples, replace=True)

    @staticmethod
    def evaluate_fit(original, generated, quantiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]):
        """评估生成的数据与原始数据的拟合度"""
        def cohens_d(x1, x2):
            n1, n2 = len(x1), len(x2)
            var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
            pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            return (np.mean(x1) - np.mean(x2)) / pooled_se

        def relative_error(x1, x2):
            return np.abs(np.mean(x1) - np.mean(x2)) / np.mean(x1)

        effect_size = cohens_d(original, generated)
        rel_error = relative_error(original, generated)
        ks_stat, _ = ks_2samp(original, generated)
        correlation = np.corrcoef(np.sort(original), np.sort(generated))[0, 1]

        orig_quant = np.quantile(original, quantiles)
        gen_quant = np.quantile(generated, quantiles)
        quant_rel_errors = np.abs(orig_quant - gen_quant) / orig_quant

        results = {
            "Cohen's d": effect_size,
            "Relative Error": rel_error,
            "KS Statistic": ks_stat,
            "Correlation": correlation,
            "Quantile Relative Errors": dict(zip([f"{q*100}th" for q in quantiles], quant_rel_errors))
        }

        excellent = (abs(effect_size) < 0.05 and rel_error < 0.05 and ks_stat < 0.1 and correlation > 0.95)
        good = (abs(effect_size) < 0.1 and rel_error < 0.15 and ks_stat < 0.15 and correlation > 0.9)
        fair = (abs(effect_size) < 0.2 and rel_error < 0.2 and ks_stat < 0.2 and correlation > 0.8)

        if excellent:
            verdict = "Excellent fit"
        elif good:
            verdict = "Good fit"
        elif fair:
            verdict = "Fair fit"
        else:
            verdict = "Poor fit"

        results["Verdict"] = verdict

        return results

    def fit_and_simulate(self, adata, n_iterations=30):
        best_simulation = None
        best_evaluation = None
        best_score = float('inf')
        best_threshold = None
        best_alpha = None
        best_beta = None

        for _ in range(n_iterations):
            self.random_state = np.random.RandomState()  # 每次迭代使用新的随机种子
            self, evaluation = self.fit(adata)
            simulated_values = self.simulate(adata.n_vars)
            final_evaluation = self.evaluate_fit(self.original_data, simulated_values)

            score = (abs(final_evaluation["Cohen's d"]) + 
                     final_evaluation["Relative Error"] + 
                     final_evaluation["KS Statistic"] + 
                     (1 - final_evaluation["Correlation"]))

            if score < best_score:
                best_score = score
                best_simulation = simulated_values
                best_evaluation = final_evaluation
                best_threshold = self.threshold
                best_alpha = self.alpha
                best_beta = self.beta

        self.threshold = best_threshold
        self.alpha = best_alpha
        self.beta = best_beta

        return best_simulation, best_evaluation

def simulate_gene_variances_advanced(adata, n_iterations=10):
    simulator = IG_VarianceSimulator()
    simulated_values, final_evaluation = simulator.fit_and_simulate(adata, n_iterations)

    result_dict = dict(zip(adata.var_names, simulated_values))

    print("\nFinal evaluation:")
    print(final_evaluation)

    # 添加更多诊断信息
    print("\nDiagnostic Information:")
    print(f"Original data mean: {np.mean(simulator.original_data)}")
    print(f"Original data variance: {np.var(simulator.original_data)}")
    print(f"Simulated data mean: {np.mean(simulated_values)}")
    print(f"Simulated data variance: {np.var(simulated_values)}")
    print(f"Best threshold: {simulator.threshold}")
    print(f"Best alpha: {simulator.alpha}")
    print(f"Best beta: {simulator.beta}")

    return result_dict, simulator.threshold, final_evaluation



def simulate_gene_average_expression(adata, pseudocount=1, n_simulations=1000):
    if sp.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = adata.X
    gene_totals = X.sum(axis=0)
    gene_totals_pseudo = gene_totals + pseudocount
    total_reads = gene_totals_pseudo.sum()
    gene_probs = gene_totals_pseudo / total_reads
    simulated_totals = np.random.multinomial(int(total_reads), gene_probs, size=n_simulations)
    average_simulated_expression = simulated_totals.mean(axis=0)
    n_cells = X.shape[0]
    average_expression = average_simulated_expression / n_cells
    return dict(zip(adata.var_names, average_expression))


