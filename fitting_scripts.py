import numpy as np
import scipy.optimize as opt


# Lattice Dimensions
Lx, Ly, Lz = 48
Lt = 64


# Auxillary functions
def check_cov_matrix(cov, v=True):
    """ 
    Is covariance matrix positive definite?
    If not, are the diagonal elements non-zero?
    """
    try:
        chol_cov = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov = np.diag(cov)
        if 0 in cov:
            raise Exception("covaraince matrix not positive definite")
    return cov


def twopt_meson(pars, t):
    """ generic two-point function for meson 'pion' """
    if not isinstance(pars, tuple):
        pars = (pars,)
    val = 0
    T = Lt/2.0
    tau = T - t
    for j in xrange(0, len(pars), 2):
        Ai, Ei = pars[j], pars[j+1]
        val += 2*Ai*np.exp(-Ei*T)*np.cosh(Ei*(tau))
    return val


def twopt_meson_jac(pars, t):
    """ jacobian of above """
    val = 0
    T = Lt/2.0
    tau = T - t
    jac = []
    for j in xrange(0, len(pars), 2):
        Ai, Ei = pars[j], pars[j+1]
        coeff = 2*np.exp(-Ei*T)
        v1 = coeff*np.cosh(Ei*tau)
        v2 = coeff*Ai*(-T*np.cosh(Ei*tau) + tau*np.sinh(Ei*tau))
        jac.append(v1)
        jac.append(v2)
    jac = np.asarray(jac)
    return jac


def twopt_meson_hes(pars, t):
    """ do it later """
    return None


def c2pt_mean_cov(c2pt):
    """ computes mean and covariance matrix of two-point correlator """
    Ncfg, Tlen = c2pt.shape
    mean = c2pt.mean(axis=0)
    cov = np.empty(shape=(Ncfg, Tlen, Tlen), dtype=c2pt.dtype)
    for n in Ncfg:
        diff = c2pt[n] - mean
        cov[n] = np.outer(diff, diff)
    cov = cov.mean(axis=0)
    cov = check_cov_matrix(cov)

    return mean, cov


def chi_square(p, c2pt, func):
    """ chi square to reduce """
    T = np.arange(c2pt.shape[1])
    mean, cov = c2pt_mean_cov(c2pt)
    diff = func(p, T) - mean
    if cov.ndim == 1:
        return sum(diff*diff/cov)
    weight = np.linalg.inv(cov)
    val = np.dot(diff, np.dot(weight, diff))
    return val


