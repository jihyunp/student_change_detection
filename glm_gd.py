"""
Gradient Descent implementation of GLM : population rate as an offset + individual intercept
Jihyun Park
"""
# __author__ = 'jihyunp'

import numpy as np
from utils import loglik_poisson, loglik_bernoulli, logit, expit

class GLM():

    def __init__(self, y, X, offset=None, alpha=0.1, family='Bernoulli_logit'):
        """

        Parameters
        ----------
        y : np.ndarray
            size (N x 1) array
        X : np.ndarray
            size (N x M) array
            ** Currently only available with intercept model
               Intercept only models -> all ones array with M = 1
        alpha : float
            Learning rate
        family : str
            'Bernoulli_logit' or
            'Poisson_log'
        """
        self.y = y
        self.X = X
        if len(X.shape) > 1:
            if X.shape[1] > 1:
                print('ERROR: Only intercept model is available!')
                exit()
        if offset is not None:
            self.offset = offset
        else:
            self.offset = np.zeros(y.shape[0])

        self.alpha = alpha
        self.family = family


    def fit(self, eps=0.001, maxiter=5000, debug=False):

        # Initialize values
        beta = -30
        delta = 10000
        niter = 0
        pt = expit(self.offset + beta)
        lam = np.exp(self.offset + beta)
        res = None

        if debug:
            print(self.family)

        if self.family == 'Bernoulli_logit':
            while abs(delta) > eps:
                if niter > maxiter:
                    break
                niter += 1
                pt = expit(self.offset + beta)
                gradient = np.sum(self.y) - np.sum(pt)
                beta_prev = beta
                beta = beta + self.alpha * gradient
                delta = beta - beta_prev
                if debug:
                    print("%f\t%f" % (beta, delta))

            ll = loglik_bernoulli(pt, self.y)
            fitted_vals = pt
            res = GLMResult(ll, fitted_vals, [beta])

        elif self.family == 'Poisson_log':
            while abs(delta) > eps:
                if niter > maxiter:
                    break
                niter += 1
                lam = np.exp(self.offset + beta)
                gradient = np.sum(self.y) - np.sum(lam)
                beta_prev = beta
                beta = beta + self.alpha * gradient
                delta = beta - beta_prev
                if debug:
                    print("%f\t%f" % (beta, delta))

            ll = loglik_poisson(lam, self.y)
            fitted_vals = lam
            res = GLMResult(ll, fitted_vals, [beta])
        return res


class GLMResult():
    def __init__(self, llf, fittedvals, params):
        self.llf = llf # Loglik
        self.fittedvalues = fittedvals
        self.params = params


if __name__ == "__main__":

    # y = np.array([1,0,1,0,0,0,1, 1, 1, 1, 1, 1, 1, 1,1])
    # y = np.zeros(10)
    # y = np.ones(10)
    # X = np.ones(y.shape[0])

    # glm_bin = GLM(y, X, family='Bernoulli_logit')
    # res = glm_bin.fit(debug=True)
    # print res.llf
    # print res.fittedvalues
    # print res.params

    y = np.array([5,5,5,5,5,5,5])
    X = np.ones(y.shape[0])

    glm_pois = GLM(y, X, family='Poisson_log')
    res = glm_pois.fit(debug=True)

    print res.llf
    print res.fittedvalues
    print res.params

