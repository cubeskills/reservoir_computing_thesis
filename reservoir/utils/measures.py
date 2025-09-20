import numpy as np
from typing import Tuple
from collections import Counter
from scipy.spatial.distance import pdist
import numba
import math
from scipy.stats import gaussian_kde
from numpy.lib.stride_tricks import as_strided
# njit much faster for the averag state entropy calculation
# (due to the fact that it is a double loop) in renyi2_entropy


class Measures:
    def __init__(self, X: np.ndarray, activations: np.ndarray = None, std: float = None):
        """X is the state history matrix with n different timesteps and m neurons"""
        self.X = X.copy()
        self.activations = activations.copy()
        self.T, self.N = X.shape
        self.activation_mean = None
        self.current_mean = None
        self.global_current_std = np.std(self.X)
        self.global_activation_std = np.std(self.activations)
        self.activation_covariance_matrix = None
        self.current_covariance_matrix = None
        self.activation_eigenvalues = None
        self.current_eigenvalues = None

    def compute_current_mean(self) -> np.ndarray:
        """Compute mean of input matrix"""
        self.current_mean = np.mean(self.X, axis=0)
        return self.current_mean
    def compute_activation_mean(self) -> np.ndarray:
        self.activation_mean = np.mean(self.activations, axis=0)

        return self.activation_mean

    def compute_current_std(self) -> np.ndarray:
        self.current_std = np.std(self.X, axis=0)
        return self.current_std
    
    def compute_activation_std(self) -> np.ndarray:
        self.activation_std = np.std(self.activations, axis=0)
        return self.activation_std

    def compute_covariance(self) -> np.ndarray:
        """Compute covariance matrix"""
        if self.activation_mean is None:
            self.compute_activation_mean()
        if self.current_mean is None:
            self.compute_current_mean()
        centered_X_current = self.X - self.current_mean
        self.current_covariance_matrix = np.cov(centered_X_current, rowvar=False)
        centered_X_activation = self.activations - self.activation_mean
        self.activation_covariance_matrix = np.cov(centered_X_activation, rowvar=False)
        return self.activation_covariance_matrix

    def compute_eigenvalues(self) -> np.ndarray:
        """Compute eigenvalues of covariance matrix"""
        if self.activation_covariance_matrix is None:
            self.compute_covariance()
        # get rid of the imaginary part
        self.activation_eigenvalues = np.linalg.eigvalsh(self.activation_covariance_matrix)
        self.current_eigenvalues = np.linalg.eigvalsh(self.current_covariance_matrix)

        return self.activation_eigenvalues

    def participation_ratio(self) -> np.ndarray:
        """Compute participation ratio, see Gao et al. 2017->A theory of multineuronal dimensionality, dynamics and1
measurement"""

        if self.activation_eigenvalues is None:
            self.compute_eigenvalues()
        participation_ratio = np.sum(self.activation_eigenvalues) ** 2 / np.sum(self.activation_eigenvalues ** 2)
        return participation_ratio
    
    
    def K(self, u: np.ndarray, sigma: float) -> np.ndarray:
        """Kernel function for renyi quadratic entropy"""
        return 1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(u ** 2) / (2 * sigma ** 2))
    
    """
    @staticmethod
    def renyi2_entropy(x: np.ndarray, sigma: float) -> float:
      
        #This is an optimized version of the implementation above, since
        #it was too slow.
        
        N = x.size
        # this is the pairwise difference matrix and is the reason
        # why this function is faster than the one above
        #D = x[:, None] - x[None, :]
        # half the operations
        D = pdist(x.reshape(-1,1), metric='euclidean')
        K = np.exp(- (D ** 2) / (2 * sigma ** 2))
        kernel_sum = K.sum() / (sigma * np.sqrt(2 * np.pi))
        return -np.log(kernel_sum / (N ** 2))
    """
    
    
    def variance_of_activation_derivatives(self) -> np.ndarray:
        """Compute variance of activation derivatives for each neuron over all time steps"""
        
        derivatives = 1 - np.tanh(self.X) ** 2
        return derivatives.var(axis=0).mean()


    def eigenvalue_spread(self) -> Tuple[float, float]:
        """Compute eigenvalue spread"""
        if self.eigenvalues is None:
            self.compute_eigenvalues()

        max_eigenvalue = float(np.max(self.eigenvalues))
        min_eigenvalue = float(self.eigenvalues[self.eigenvalues > 0.0].min())
        return max_eigenvalue / min_eigenvalue
    
    def kernel_rank(self, tol: float = 1e-8) -> int:
        """Compute kernel rank"""
        
        r = int(np.linalg.matrix_rank(self.X, tol=tol))
        return r, r/ self.T
        
    
    
    def mutual_information(self, u: np.ndarray, v: np.ndarray, uv: np.ndarray) -> float:
        """Compute mutual information between two distributions"""
        return np.sum(uv * np.log(uv / u / v))
    
    def lais_kde(self, time_series: np.ndarray, k: int = 2, sigma: float = None) -> float:
        """computing local active information storage via kernel density estimation"""
        T = len(time_series)
        N = T-k
        if N <= 0:
            return 0.0
        
        U = np.vstack([time_series[i : i + k] for i in range(N)]) 
        V = time_series[k:k+N].reshape(N, 1)
        U = U.T
        V = V.T
        UV = np.vstack([U, V])
        
        # estimating the kernel density
        kde_uv = gaussian_kde(UV, bw_method=sigma)
        kde_u = gaussian_kde(U, bw_method=sigma)
        kde_v = gaussian_kde(V, bw_method=sigma)
        # computing the probabilities
        prob_uv = kde_uv(UV)
        prob_u = kde_u(U)
        prob_v = kde_v(V)

        ais = np.sum(prob_uv * np.log2(prob_uv / (prob_u * prob_v)))
        return ais 
    
    
    def lais(self,time_series: np.ndarray, k: int = 2) -> float:

        """compute  local active information storage, via 
        estimating the probability distributions of the (signed) history of the neuron
        and then computing the mutual information between the two
        distributions.
        """
        """
        T = len(time_series)
        N = T-k
        if N <= 0:
            return 0.0
        
        uv = []
        for i in range(N):
            u = tuple(time_series[i : i + k])
            v = time_series[i+k]
            uv.append(u + (v,))
        # 
        joint_counts = Counter(uv)
        history_counts = Counter(mutual[:-1] for mutual in uv)
        value_counts = Counter(mutual[-1] for mutual in uv)
        # counting occurences of the patterns
        patterns = list(joint_counts.keys())
        count_joint = np.array([joint_counts[p] for p in patterns],dtype=float)
        count_history = np.array([history_counts[p[:-1]] for p in patterns],dtype=float)
        count_value = np.array([value_counts[p[-1]] for p in patterns],dtype=float)
        # calculating probbabilities
        prob_joint = count_joint / N
        prob_history = count_history / N
        prob_value = count_value / N

        return float(np.sum(prob_joint * np.log2(prob_joint / (prob_history * prob_value))))
        """
        T = len(time_series)
        N = T - k
        if N <= 0:
            return 0.0
        ts = ((time_series + 1) / 2).astype(np.int8)
        strided_hist = as_strided(ts, shape=(N, k), strides=(ts.strides[0], ts.strides[0]))
        powers_of_2 = (2 ** np.arange(k)).reshape(1, -1)
        hist = (strided_hist * powers_of_2).sum(axis=1)
        targets = ts[k:]
        k_range = 2 ** k
        # probability distributions
        p_hist = np.bincount(hist, minlength=k_range)/N
        p_target = np.bincount(targets, minlength=2) / N
        joint_ = targets + hist * 2
        p_joint = np.bincount(joint_, minlength=2 * k_range) / N
        # reshape
        p_joint = p_joint.reshape((k_range, 2))

        denominator = p_hist[:,np.newaxis] * p_target[np.newaxis,:]

        valid_mask = p_joint > 0
        fraction = p_joint[valid_mask] / denominator[valid_mask]
        ais = np.sum(p_joint[valid_mask] * np.log2(fraction))

        
        return ais
        



    def ais(self, k: int = 2) -> float:
        """Compute average active information storage"""
        T,N = self.activations.shape
        if k <= 0 or k > self.T:
            raise ValueError("history must be greater than 1 and less than T")
        ais_values = []
        for neuron in range(N):
            time_series = np.sign(self.activations[:, neuron])
            time_series[time_series == 0] = 1
            ais_values.append(self.lais(time_series, k=k))

        return np.mean(ais_values)
  
    def saturation_mean(self,threshold: float = 1.5) -> float:
        """Compute the mean saturation of the activations"""
        T,N = self.X.shape
        if self.X is None:
            raise ValueError("Activations are not set.")
        abs_currents = np.abs(self.X)
        # count how many activations are above the threshold
        saturation_count = np.sum(abs_currents > threshold, axis=1)
        #print(saturation_count)
        # compute the mean saturation
        mean_saturation = saturation_count / N
        return np.mean(mean_saturation)
    
    def saturation_std(self, threshold: float = 1.5) -> float:
        """Compute the standard deviation of the saturation of the activations"""
        T,N = self.X.shape
        if self.X is None:
            raise ValueError("Activations are not set.")
        abs_currents = np.abs(self.X)
        # count how many activations are above the threshold
        saturation_count = np.sum(abs_currents > threshold, axis=1)
        # compute the standard deviation of saturation
        std_saturation = np.std(saturation_count / N)
        return std_saturation
    
    def mean_correlation(self) -> float:
        """Compute the mean correlation between neuron activations"""
        
        corr_matrix = np.corrcoef(self.activations, rowvar=False)
        N = corr_matrix.shape[0]
        # remove diagonal elements as they are all 1 (we want
        # cross correlations not autocorrelations)
        positive_correlations = np.abs(corr_matrix) - np.eye(N)
        return np.sum(positive_correlations) / (N * (N - 1))
    
    def pairwise_te(self, time_series1: np.ndarray, time_series2: np.ndarray, k: int = 2) -> float:
        """
        T = len(time_series1)
        N = T-k

        if N <= 0:
            return 0.0
        uvt = [
        (tuple(time_series1[i:i+k]),
         tuple(time_series2[i:i+k]),
         time_series2[i+k])
        for i in range(N)
    ]
        # Counter is used for counting the occurences of the patterns
        # it returns a dict the same way is in ais
        count_uvt = Counter(uvt)
        count_uv = Counter((u,v) for u,v,t in uvt)
        count_vt = Counter((v,t) for u,v,t in uvt)
        count_v = Counter(v for u,v,t in uvt)

        te = 0.0
        for (u,v,t), c_uvt in count_uvt.items():
            p_uvt = c_uvt / N
            p_uv = count_uv[(u,v)] / N
            p_vt = count_vt[(v,t)] / N
            p_v = count_v[v] / N
            te += p_uvt * np.log2((p_uvt/p_uv) / (p_vt / p_v))
        """
        T = len(time_series1)
        N = T - k
        if N <= 0:
            return 0.0
        ts1 = ((time_series1 +1)/2).astype(np.int8)
        ts2 = ((time_series2 +1)/2).astype(np.int8)
        powers_of_2 = (2**np.arange(k)).reshape(1, -1)

        # sliding window to create the history
        strided_u = as_strided(ts1, shape=(N, k), strides=(ts1.strides[0], ts1.strides[0]))
        strided_v = as_strided(ts2, shape=(N, k), strides=(ts2.strides[0], ts2.strides[0]))

        # count appearances and probabilities
        u = (strided_u * powers_of_2).sum(axis=1)
        v = (strided_v * powers_of_2).sum(axis=1)
        t = ts2[k:]
        k_range = 2**k
        p_uvt = np.bincount(t + v * 2 + u * 2 * k_range, minlength=2 * k_range * k_range) / N
        p_uv = np.bincount(v + u * k_range, minlength=k_range * k_range) / N
        p_vt = np.bincount(t + v * 2, minlength=2 * k_range) / N
        p_v = np.bincount(v, minlength=k_range) / N
        # reshape
        p_uvt = p_uvt.reshape((k_range, k_range, 2))
        p_uv = p_uv.reshape((k_range, k_range,1))
        p_vt = p_vt.reshape((k_range, 2))
        p_v = p_v.reshape((k_range, 1))

        numerator = p_uvt * p_v
        denominator = p_uv * p_vt
        valid_mask = (numerator > 0) & (denominator > 0)
        log_term = np.log2(numerator[valid_mask] / denominator[valid_mask])
        te = np.sum(p_uvt[valid_mask] * log_term)
        
        return te
    
    def te(self, num_pairs: int = 100, k: int = 2, seed: int = 42) -> float:
        """ uses pairwise te to compute the average over a sample of neuron pairs"""
        T, N = self.activations.shape
        max_pairs = N*(N-1)
        if num_pairs < 1 or num_pairs > max_pairs:
            raise ValueError(f"num_pairs must be between 1 and {max_pairs}, but got {num_pairs}.")
        
        rng = np.random.RandomState(seed=seed)
        sampled_pairs = set()
        while len(sampled_pairs) < num_pairs:

            i = rng.randint(0, N)
            j = rng.randint(0, N)
            if i != j:
                sampled_pairs.add((i, j))
        
        te_values = []
        for i, j in sampled_pairs:

            ts_i = np.sign(self.activations[:, i])
            ts_i[ts_i == 0] = 1
            ts_j = np.sign(self.activations[:, j])
            ts_j[ts_j == 0] = 1
            te_ij = self.pairwise_te(ts_i, ts_j, k=k)
            te_values.append(te_ij)
    
        return float(np.mean(te_values))

    
    def renyi2_entropy(self, x: np.ndarray, sigma: float) -> float:
        """Compute only the renyi 2 entropy
        see ozturk et al. 2007
        """

        N = x.size
        # this is the pairwise difference matrix and is the reason
        # why this function is faster than the one above
        D = x[:, None] - x[None, :]
        K = np.exp(- (D ** 2) / (2 * sigma ** 2))
        kernel_sum = K.sum() / (sigma * np.sqrt(2 * np.pi))
        return -np.log(kernel_sum / (N ** 2))
          
    
        
    def instantaneous_state_entropy(self, x_t: np.ndarray, sigma: float = 0.3) -> float:
        """compute the instantaneous state entropy of a single time step"""
        std = self.global_activation_std
        if std == 0:
            return 0.0
        
        sigma = sigma * std
        return self.renyi2_entropy(x_t, sigma)

    def average_state_entropy(self, X: np.ndarray=None, sigma: float = 0.3) -> float:
        
        """Compute average state entropy"""
        if X is None:
            X = self.activations

        ents = np.zeros(self.T)
        for t in range(self.T):
            x_t = self.X[t]
            ents[t] = self.instantaneous_state_entropy(x_t, sigma=sigma)

        return np.mean(ents)
    
    def scaling_ratio(self) -> float:
        """
        Computes the ratio of recurrent-driven dynamics to input-driven dynamics.
        
        """
        
        autocorrelations = []
        for n in range(self.N):
            time_series = self.X[:, n]
            if np.std(time_series) > 1e-9:
                corr_matrix = np.corrcoef(time_series[:-1], time_series[1:])
                autocorrelations.append(corr_matrix[0, 1])
            else:
                autocorrelations.append(0.0)
        
        mean_abs_autocorr = np.mean(np.abs(autocorrelations))
        ratio = mean_abs_autocorr / (1.0 - mean_abs_autocorr + 1e-9)
        
        return ratio
    def average_state_vector_length(self) -> float:
        """Compute the average length of the difference vector between 2 consequtive neuron state vectors"""
        vector_lengths = []
        for t in range(1, self.T):
            v1 = self.X[t - 1]
            v2 = self.X[t]
            diff_vector = v2 - v1
            vector_lengths.append(np.linalg.norm(diff_vector))
        return np.mean(vector_lengths)

    def current_variance(self) -> float:

        return np.var(self.X, axis=1).mean()
    
    def scale_recurrency_ratio(self) -> float:
        vector_diffs = []
        """calculate for every 2 consecutive time steps the angle between the neuron-state vectors"""
        for t in range(1, self.T):
            v1 = self.X[t - 1]
            v2 = self.X[t]
            if np.linalg.norm(v1) > 1e-9 and np.linalg.norm(v2) > 1e-9:
                cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                vector_diffs.append(cos_theta)
            else:
                vector_diffs.append(0.0)
        vector_diffs = np.array(vector_diffs)
        mean_angle = np.mean(np.abs(vector_diffs))
        return mean_angle

    def dynamic_range_ratio(self) -> float:
        
        var_pre_activation = np.var(self.X)
        var_post_activation = np.var(self.activations)

        if var_pre_activation < 1e-12:
            return 0.0
            
        return var_post_activation / var_pre_activation

    def array_entropy(self, X:np.ndarray=None) -> float:

        """ compute entropy of input array"""
        if X is None:
            X = self.X
        return -np.sum(X * np.log2(X + 1e-10)) / np.log2(X.shape[0])
    
    def entropy_rate(self, X: np.ndarray=None) -> float: 
        pass

    def excess_entropy(self, X: np.ndarray=None) -> float:
        """Compute excess entropy"""
        pass

