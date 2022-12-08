


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn import datasets
from base import BaseGMM, _expectation_dirichlet


class Dirichlet_GMM(BaseGMM):
    def __init__(self, n_components, max_iter=100, tol=1e-3, random_state=None, prob_norm_init=None, alpha_init=None):
        super().__init__(n_components, max_iter, tol, random_state, prob_norm_init)
        self.alpha_init = alpha_init

    def _initialize(self, X, resp):
        self.alpha = np.full(shape=self.n_components, fill_value=1./self.n_components) if self.alpha_init is None else self.alpha_init
        self.weights_, self.means_, self.covariances_ = self._estimate_gaussian_parameters(X, resp)
        self.weights_ /= np.sum(self.weights_)

    def _e_step(self, X):
        
        n_samples, _ = X.shape
        nk = self.weights_ * n_samples
        self.alpha += nk

        
        likelihood = np.zeros(shape=(n_samples, self.n_components))
        for k in range(self.n_components):
            distribution = multivariate_normal(mean=self.means_[k], cov=self.covariances_[k])
            likelihood[:, k] = _expectation_dirichlet(self.alpha, k) + np.log(distribution.pdf(X))

        numerator = np.exp(likelihood)
        denominator = np.sum(numerator, axis=1)[:, np.newaxis]
        return numerator / denominator

    def _m_step(self, X, resp):
        _, self.means_, self.covariances_ = self._estimate_gaussian_parameters(X, resp)



def main():

    X, y, centers = datasets.make_blobs(n_samples=50000, n_features=2, centers=3, center_box=(-5,5), return_centers=True, random_state=0)   

    fig, ax = plt.subplots()
    ax.scatter(X[np.where(y==0), 0], X[np.where(y==0), 1], s=5, color='r', label=f'x0={centers[0][0]:.2f},x1={centers[0][1]:.2f}')
    ax.scatter(X[np.where(y==1), 0], X[np.where(y==1), 1], s=5, color='b', label=f'x0={centers[1][0]:.2f},x1={centers[1][1]:.2f}')
    ax.scatter(X[np.where(y==2), 0], X[np.where(y==2), 1], s=5, color='g', label=f'x0={centers[2][0]:.2f},x1={centers[2][1]:.2f}')
    ax.legend()
    plt.tight_layout()
    # fig.savefig('trainingdata.png')

    gmm = Dirichlet_GMM(n_components=3, max_iter=1000)
    gmm.fit(X)
    print(centers)
    print(gmm.means_)

if __name__ == '__main__':
    main()