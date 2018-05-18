import functools as ft
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import lat_stats as lstats


# Making Effective Mass Plots for Two-Point Function
def func(meff, t, ratio):
    """ main function """
    Nt = 64.0
    tau = t - Nt/2.0
    func_ratio = np.cosh(meff*(tau+1))/np.cosh(meff*tau)
    return func_ratio - ratio


def jac(meff, t, ratio):
    """ jacobian """
    Nt = 64.0
    tau = t - Nt/2.0
    A = (tau+1)*np.sinh(meff*(tau+1))/np.cosh(meff*tau)
    B = -tau*np.sinh(meff*tau)*np.cosh(meff*(tau+1))/(np.cosh(meff*tau)**2)
    return A + B


def hess(meff, t, ratio):
    """ hessian """
    Nt = 64.0
    tau = t - Nt/2.0
    Aa = 2*(np.sinh(meff*tau)**2)*np.cosh(meff*(tau+1))/(np.cosh(meff*tau)**3)
    Ab = -np.cosh(meff*(tau+1))/np.cosh(meff*tau)
    A = Aa + Ab
    B = -2*(tau+1)*np.sinh(meff*tau)*np.sinh(meff*(tau+1))/(np.cosh(meff*tau)**2)
    C = ((tau+1)**2)*np.cosh(meff*(tau+1))/np.cosh(meff*tau)
    return A*tau**2 + B*tau + C


# Getting initial guess
def dispers(psq):
    Nx = 48.
    mpisq = (0.0913)**2
    psq *= (2*np.pi/Nx)
    return np.sqrt(psq + mpisq)


def newton_meff(func, Dat, guess, fprime=None, fprime2=None, **kwargs):
    """ gives me effective mass plot as function of time: newton """
    jk_smp = Dat.shape[0]
    T = np.arange(Dat.shape[-1])
    ratio = np.roll(Dat, 1, axis=-1).mean(axis=0)/Dat.mean(axis=0)
    # cv == central value
    cv_arglist = [(t, ratio[t_iter]) for t_iter, t in enumerate(T)]
    cv = [
        opt.newton(func, guess, args=cv_arg,
                   fprime=fprime, fprime2=fprime2)
        for cv_arg in cv_arglist
    ]
    return cv


def newton_meff_mean_var(func, Dat, guess, fprime=None, fprime2=None, **kwargs):
    """ 
    gives me effective mass plot as function of time with jackknife error: newton
    """
    jk_smp = Dat.shape[0]
    T = np.arange(Dat.shape[-1])
    Blocks = lstats.jk_blocks(Dat, jk_smp, axis=0)
    cv = newton_meff(func, Dat, guess, fprime, fprime2, **kwargs)
    cvblk = np.array([
        newton_meff(func, blk, guess, fprime, fprime2, **kwargs)
        for blk in Blocks
    ])
    diff = np.array([
        pow(cv - cvb, 2) for cvb in cvblk
    ])
    err = ((jk_smp - 1.0)/jk_smp)*diff.sum(axis=0)
    err = np.sqrt(err)
    return cv, err


def bisect_meff(func, Dat, a, b):
    """ gives me effective mass plot as a function of time: brentq """
    jk_smp = Dat.shape[0]
    T = np.arange(Dat.shape[-1])
    ratio = np.roll(Dat, 1, axis=-1).mean(axis=0)/Dat.mean(axis=0)
    cv_arglist = [(t, ratio[t_iter]) for t_iter, t in enumerate(T)]
    cv = [
        opt.brentq(func, a, b, args=cv_arg)
        for cv_arg in cv_arglist
    ]
    return cv


def bisect_meff_mean_var(func, Dat, a, b):
    """
    gives me effective mass plot as function of time with jackknife error: brentq
    """
    jk_smp = Dat.shape[0]
    T = np.arange(Dat.shape[-1])
    Blocks = lstats.jk_blocks(Dat, jk_smp, axis=0)
    cv = bisect_meff(func, Dat, a, b)
    cvblk = np.array([bisect_meff(func, blk, a, b) for blk in Blocks])
    diff = np.array([pow(cv - cvb, 2) for cvb in cvblk])
    err = ((jk_smp - 1.0)/jk_smp)*diff.sum(axis=0)
    err = np.sqrt(err)
    return cv, err
