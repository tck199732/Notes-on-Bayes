import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import digamma
from abc import ABC, abstractmethod

def _expectation_dirichlet(alpha, k):
    return digamma(alpha[k]) - digamma(np.sum(alpha))

class BaseGMM(ABC):
    def __init__(self, n_components, max_iter, tol, random_state, prob_norm_init):
        self.n_components = n_components
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol
        self.prob_norm_init = prob_norm_init

        self._fitted = False

    def fit(self, X):
        self._initialize_parameters(X, self.random_state)
        tol = -np.Inf
        iter = 0
        while tol < self.tol and iter < self.max_iter:
            resp = self._e_step(X)
            self._m_step(X, resp)
            # update tol here
            iter += 1
        
        self.fitted = True

    def _initialize_parameters(self, X, random_state):

        self.prob_norm = np.full(shape=self.n_components, fill_value=1./self.n_components) if self.prob_norm_init is None else self.prob_norm_init

        n_samples, _ = X.shape
        resp = np.zeros(shape=(n_samples, self.n_components))

        if random_state is None:
            random_state = 0

        np.random.seed(random_state)
        for n in range(n_samples):
            k = np.random.randint(0, self.n_components)
            resp[n][k] = 1
        
        self._initialize(X, resp)

    @abstractmethod
    def _initialize(self, X, resp):
        pass

    @abstractmethod 
    def _e_step(self, X):
        pass 

    @abstractmethod
    def _m_step(self, X, resp):
        pass

    def _estimate_gaussian_parameters(self, X, resp):
        _, n_features = X.shape
        nk = np.sum(resp, axis=0)
        means = np.dot(resp.T, X) / nk[:, np.newaxis]
        var = np.empty(shape=(self.n_components, n_features, n_features))
        for k in range(self.n_components):
            diff = X - means[k]
            var[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
        
        return nk, means, var
    
    def predict_proba(self, X):
        self._check_is_fitted()
        return self._e_step()

    def predict(self, X):
        self._check_is_fitted()
        resp = self.predict_proba(X)
        return resp.argmax(axis=1)
        

    def _check_is_fitted(self):
        if not self._fitted:
            raise ValueError('model not trained.')
        