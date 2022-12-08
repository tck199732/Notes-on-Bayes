
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn import datasets
from base import BaseGMM

class GMM(BaseGMM):

    def __init__(self, n_components, max_iter, tol=1e-3, random_state=None, prob_norm_init=None):
        super().__init__(n_components, max_iter, tol, random_state)
        self.prob_norm_init = prob_norm_init

    def _initialize(self, X, resp):

        self.weights_, self.means_, self.covariances_ = self._estimate_gaussian_parameters(X, resp)
        self.weights_ /= np.sum(self.weights_)

    def _e_step(self, X):
        n_samples, _ = X.shape
        
        likelihood = np.zeros(shape=(n_samples, self.n_components))
        for k in range(self.n_components):
            distribution = multivariate_normal(mean=self.means_[k], cov=self.covariances_[k])
            likelihood[:, k] = distribution.pdf(X)
        
        numerator = likelihood * self.prob_norm
        denominator = numerator.sum(axis=1)[:, np.newaxis]
        return numerator / denominator
        
    def _m_step(self, X, resp):

        self.weights_, self.means_, self.covariances_ = self._estimate_gaussian_parameters(X, resp)
        self.weights_ /= np.sum(self.weights_)



def main():

    X, y, centers = datasets.make_blobs(n_samples=50000, n_features=2, centers=3, center_box=(-5,5), return_centers=True, random_state=0)   

    fig, ax = plt.subplots()
    ax.scatter(X[np.where(y==0), 0], X[np.where(y==0), 1], s=5, color='r', label=f'x0={centers[0][0]:.2f},x1={centers[0][1]:.2f}')
    ax.scatter(X[np.where(y==1), 0], X[np.where(y==1), 1], s=5, color='b', label=f'x0={centers[1][0]:.2f},x1={centers[1][1]:.2f}')
    ax.scatter(X[np.where(y==2), 0], X[np.where(y==2), 1], s=5, color='g', label=f'x0={centers[2][0]:.2f},x1={centers[2][1]:.2f}')
    ax.legend()
    plt.tight_layout()
    # fig.savefig('trainingdata.png')

    gmm = GMM(n_components=3, max_iter=1000)
    gmm.fit(X)
    print(centers)
    print(gmm.means_)

    # dpgmm = GaussianMixture(n_components=3, max_iter=1000)
    # dpgmm = BayesianGaussianMixture(n_components=3, max_iter=1000)
    # dpgmm.fit(X)
    # print(dpgmm.means_)

if __name__ == '__main__':
    main()