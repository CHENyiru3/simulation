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


import numpy as np
import scipy.sparse as sp
from scipy.stats import ks_2samp
from scipy.optimize import minimize
from scipy.special import gamma

import numpy as np
import scipy.sparse as sp
from scipy.stats import ks_2samp, invgamma, gamma, skew
from scipy.optimize import minimize
from scipy.special import gamma as gamma_func
import warnings
warnings.filterwarnings('ignore')

class GIG_VarianceSimulator:
    def __init__(self):
        self.alpha = None
        self.beta = None
        self.theta = None
        self.k = None
        self.lambda_ = None
        self.threshold = None
        self.tail_data = None
        self.original_order = None
        self.original_data = None
        self.random_state = None
        self.model_type = None  # 'IG' or 'GIG'

    def generalized_gamma(self, alpha, k, lambda_):
        """数值稳定的广义伽马函数计算"""
        try:
            log_gamma = np.log(gamma_func(alpha)) - lambda_ * np.log(1 + k)
            return np.exp(log_gamma)
        except:
            return gamma_func(alpha)

    def gig_pdf(self, x, alpha, beta, theta, k, lambda_):
        """数值稳定的PDF计算"""
        try:
            log_pdf = (np.log(beta) + 
                      alpha*beta*np.log(theta) - 
                      (alpha*beta + 1)*np.log(x) - 
                      lambda_*np.log((theta/x)**beta + k) - 
                      (theta/x)**beta - 
                      np.log(self.generalized_gamma(alpha, k, lambda_)))
            return np.exp(log_pdf)
        except:
            return np.zeros_like(x)

    def extract_data(self, X):
        variances = np.var(X, axis=0, ddof=1)
        variances = np.nan_to_num(variances, nan=np.nanmean(variances))
        variances = np.maximum(variances, 1e-10)
        return variances

    def get_initial_params(self, data):
        """智能初始参数估计"""
        mean = np.mean(data)
        var = np.var(data)
        skewness = skew(data)
        
        # 基于统计量估计初始参数
        alpha_init = max(0.1, (2 + skewness**2) / abs(skewness))
        beta_init = 1.0
        theta_init = mean * (alpha_init - 1) if alpha_init > 1 else mean
        k_init = 1.0
        lambda_init = 0.5
        
        return [alpha_init, beta_init, theta_init, k_init, lambda_init]

    def fit_simple_ig(self, data):
        """拟合简单逆伽马分布"""
        try:
            params = invgamma.fit(data, floc=0)
            return {'alpha': params[0], 'scale': params[2]}
        except:
            return {'alpha': 1.0, 'scale': np.mean(data)}

    def calculate_aic(self, log_likelihood, n_params):
        """计算AIC"""
        return 2 * n_params - 2 * log_likelihood

    def fit_gig(self, data, initial_guess=None):
        """改进的GIG拟合"""
        def negative_log_likelihood(params):
            alpha, beta, theta, k, lambda_ = params
            
            # 添加正则化项
            penalty = 0.01 * (alpha**2 + beta**2 + k**2 + lambda_**2)
            
            try:
                pdf_values = self.gig_pdf(data, alpha, beta, theta, k, lambda_)
                pdf_values = np.maximum(pdf_values, 1e-300)
                return -np.sum(np.log(pdf_values)) + penalty
            except:
                return np.inf

        bounds = [
            (0.1, 20),    # alpha
            (0.1, 5),     # beta
            (0.1, None),  # theta
            (0.1, 10),    # k
            (0, 5)        # lambda
        ]

        if initial_guess is None:
            initial_guess = self.get_initial_params(data)

        best_fit = None
        best_likelihood = np.inf
        
        # 多次尝试不同初始值
        initial_guesses = [
            initial_guess,
            [1.0, 1.0, np.mean(data), 1.0, 1.0],
            [2.0, 0.5, np.median(data), 0.5, 0.1]
        ]

        for guess in initial_guesses:
            try:
                result = minimize(negative_log_likelihood, 
                                guess,
                                bounds=bounds,
                                method='L-BFGS-B')
                if result.success and result.fun < best_likelihood:
                    best_likelihood = result.fun
                    best_fit = result.x
            except:
                continue

        if best_fit is None:
            raise ValueError("GIG fitting failed for all initial guesses")
            
        return best_fit, best_likelihood

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

            # 先尝试简单IG拟合
            ig_params = self.fit_simple_ig(sorted_data)
            
            thresholds = [95,96,97,98,99,99.5]
            best_score = float('inf')
            best_evaluation = None
            
            for percentile in thresholds:
                current_threshold = np.percentile(sorted_data, percentile)
                main_data = sorted_data[sorted_data <= current_threshold]
                tail_data = sorted_data[sorted_data > current_threshold]

                if len(main_data) == 0 or len(tail_data) == 0:
                    continue

                # 尝试GIG拟合
                try:
                    gig_params, gig_likelihood = self.fit_gig(main_data)
                    self.alpha, self.beta, self.theta, self.k, self.lambda_ = gig_params
                    
                    # 计算AIC
                    gig_aic = self.calculate_aic(gig_likelihood, 5)
                    ig_aic = self.calculate_aic(-len(main_data)*np.log(invgamma.pdf(main_data, 
                                              ig_params['alpha'], 
                                              scale=ig_params['scale'])).sum(), 2)
                    
                    # 选择更好的模型
                    if ig_aic < gig_aic:
                        self.model_type = 'IG'
                        self.alpha = ig_params['alpha']
                        self.theta = ig_params['scale']
                    else:
                        self.model_type = 'GIG'

                    n_main = len(main_data)
                    n_tail = len(tail_data)

                    # 根据选择的模型生成主体部分数据
                    if self.model_type == 'IG':
                        new_main = invgamma.rvs(self.alpha, scale=self.theta, 
                                              size=n_main, random_state=self.random_state)
                    else:
                        new_main = self.simulate_gig(n_main)

                    # 处理尾部数据
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
                    new_samples = np.clip(new_samples, np.min(self.original_data), 
                                        np.max(self.original_data))

                    evaluation = self.evaluate_fit(self.original_data, new_samples)

                    score = (abs(evaluation["Cohen's d"]) + 
                            evaluation["Relative Error"] + 
                            evaluation["KS Statistic"] + 
                            (1 - evaluation["Correlation"]))

                    if score < best_score:
                        best_score = score
                        self.threshold = percentile
                        best_evaluation = evaluation
                        self.tail_data = tail_data

                except Exception as e:
                    print(f"Fitting at threshold {percentile} failed: {str(e)}")
                    continue

            if best_evaluation is None:
                raise ValueError("No valid fit found")

            return self, best_evaluation

        except Exception as e:
            print(f"Fitting failed: {str(e)}")
            # Fallback to simple IG
            ig_params = self.fit_simple_ig(self.original_data)
            self.model_type = 'IG'
            self.alpha = ig_params['alpha']
            self.theta = ig_params['scale']
            return self, {"Verdict": "Fallback to single component IG"}

    def simulate_gig(self, n_samples):
        def adaptive_proposal(size):
            if self.k < 1 and self.lambda_ < 1:
                return invgamma.rvs(self.alpha, scale=self.theta, size=size)
            else:
                return gamma.rvs(self.alpha, scale=1/self.theta, size=size)

        def calculate_acceptance_ratio(x):
            if self.model_type == 'IG':
                proposal_pdf = invgamma.pdf(x, self.alpha, scale=self.theta)
                target_pdf = proposal_pdf
            else:
                proposal_pdf = invgamma.pdf(x, self.alpha, scale=self.theta)
                target_pdf = self.gig_pdf(x, self.alpha, self.beta, 
                                        self.theta, self.k, self.lambda_)
            return target_pdf / np.maximum(proposal_pdf, 1e-300)

        accepted_samples = []
        max_attempts = 100
        attempt = 0

        while len(accepted_samples) < n_samples and attempt < max_attempts:
            proposed = adaptive_proposal(n_samples * 2)
            acceptance_ratio = calculate_acceptance_ratio(proposed)
            acceptance_prob = acceptance_ratio / np.max(acceptance_ratio)
            
            accepted = proposed[self.random_state.random(len(proposed)) < acceptance_prob]
            accepted_samples.extend(accepted[:n_samples-len(accepted_samples)])
            attempt += 1

        if len(accepted_samples) < n_samples:
            # Fallback to proposal distribution
            remaining = n_samples - len(accepted_samples)
            accepted_samples.extend(adaptive_proposal(remaining))

        return np.array(accepted_samples[:n_samples])

    def simulate(self, n_samples):
        try:
            if self.threshold is None:
                print("Warning: Threshold is None, using default value of 95.")
                self.threshold = 95

            n_main = int(n_samples * self.threshold / 100)
            n_tail = n_samples - n_main

            # 根据选择的模型生成主体部分
            if self.model_type == 'IG':
                new_main = invgamma.rvs(self.alpha, scale=self.theta, 
                                      size=n_main, random_state=self.random_state)
            else:
                new_main = self.simulate_gig(n_main)

            # 处理尾部
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
            new_samples = np.clip(new_samples, np.min(self.original_data), 
                                np.max(self.original_data))
            new_samples = np.sort(new_samples)

            simulated_data = np.zeros_like(new_samples)
            simulated_data[self.original_order] = new_samples

            return simulated_data

        except Exception as e:
            print(f"Simulation failed: {str(e)}")
            return self.random_state.choice(self.original_data, size=n_samples, replace=True)

    @staticmethod
    def evaluate_fit(original, generated, quantiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]):
        """评估拟合质量"""
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

        excellent = (abs(effect_size) < 0.05 and rel_error < 0.05 and 
                    ks_stat < 0.1 and correlation > 0.95)
        good = (abs(effect_size) < 0.1 and rel_error < 0.15 and 
                ks_stat < 0.15 and correlation > 0.9)
        fair = (abs(effect_size) < 0.2 and rel_error < 0.2 and 
                ks_stat < 0.2 and correlation > 0.8)

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
        thresholds = [95, 97.5, 99, 99.5]
        best_overall_simulation = None
        best_overall_evaluation = None
        best_overall_score = float('inf')
        best_overall_threshold = None
        best_overall_params = None

        # 对每个阈值进行独立的优化
        for threshold in thresholds:
            print(f"Optimizing for threshold {threshold}...")
            
            # 针对当前阈值进行多次迭代优化
            best_threshold_simulation = None
            best_threshold_evaluation = None
            best_threshold_score = float('inf')
            best_threshold_params = None

            for iteration in range(n_iterations):
                self.random_state = np.random.RandomState(iteration)
                self.threshold = threshold  # 固定当前阈值
                
                try:
                    self, evaluation = self.fit(adata)
                    simulated_values = self.simulate(adata.n_vars)
                    final_evaluation = self.evaluate_fit(self.original_data, simulated_values)

                    score = (abs(final_evaluation["Cohen's d"]) + 
                            final_evaluation["Relative Error"] + 
                            final_evaluation["KS Statistic"] + 
                            (1 - final_evaluation["Correlation"]))

                    # 更新当前阈值的最佳结果
                    if score < best_threshold_score:
                        best_threshold_score = score
                        best_threshold_simulation = simulated_values
                        best_threshold_evaluation = final_evaluation
                        best_threshold_params = {
                            'threshold': self.threshold,
                            'alpha': self.alpha,
                            'beta': self.beta if hasattr(self, 'beta') else None,
                            'theta': self.theta,
                            'k': self.k if hasattr(self, 'k') else None,
                            'lambda_': self.lambda_ if hasattr(self, 'lambda_') else None,
                            'model_type': self.model_type
                        }

                except Exception as e:
                    print(f"Iteration {iteration} for threshold {threshold} failed: {str(e)}")
                    continue

            # 比较当前阈值的最佳结果与全局最佳结果
            if best_threshold_score < best_overall_score:
                best_overall_score = best_threshold_score
                best_overall_simulation = best_threshold_simulation
                best_overall_evaluation = best_threshold_evaluation
                best_overall_threshold = threshold
                best_overall_params = best_threshold_params

            print(f"Best score for threshold {threshold}: {best_threshold_score}")
            print(f"Best evaluation for threshold {threshold}: {best_threshold_evaluation}")

        # 设置全局最佳参数
        self.threshold = best_overall_threshold
        self.model_type = best_overall_params['model_type']
        self.alpha = best_overall_params['alpha']
        self.theta = best_overall_params['theta']
        
        if self.model_type == 'GIG':
            self.beta = best_overall_params['beta']
            self.k = best_overall_params['k']
            self.lambda_ = best_overall_params['lambda_']

        print(f"\nFinal selected threshold: {best_overall_threshold}")
        print(f"Final model type: {self.model_type}")
        print(f"Final evaluation: {best_overall_evaluation}")

        return best_overall_simulation, best_overall_evaluation

def simulate_gene_variances_advanced(adata, n_iterations=10):
    """主函数：模拟基因方差"""
    simulator = GIG_VarianceSimulator()
    try:
        simulated_values, final_evaluation = simulator.fit_and_simulate(adata, n_iterations)
        result_dict = dict(zip(adata.var_names, simulated_values))
        
        print(f"\nDetailed Results:")
        print(f"Selected threshold: {simulator.threshold}")
        print(f"Model type: {simulator.model_type}")
        if simulator.model_type == 'GIG':
            print(f"GIG parameters - alpha: {simulator.alpha:.3f}, beta: {simulator.beta:.3f}, "
                  f"theta: {simulator.theta:.3f}, k: {simulator.k:.3f}, lambda: {simulator.lambda_:.3f}")
        else:
            print(f"IG parameters - alpha: {simulator.alpha:.3f}, theta: {simulator.theta:.3f}")
        
        return result_dict, simulator.threshold, final_evaluation
        
    except Exception as e:
        print(f"Simulation failed: {str(e)}")
        raise
    
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


