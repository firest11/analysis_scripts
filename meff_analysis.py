import functools as ft
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import lat_stats as lstats


# Auxillary Functions
def match_lists(my_lists, match=True):
    """ returns domain and range of list """
    func_table = {True: np.intersect1d, False: np.union1d}
    domain_list = [my_list[0] for my_list in my_lists]
    domain = ft.reduce(func_table[match], domain_list)
    image = np.empty(shape=(len(my_lists), len(domain)),
                     dtype=my_lists[0][1].dtype)
    for j, my_list in enumerate(my_lists):
        for k, dm in enumerate(domain):
            for tval, ival in zip(my_list[0], my_list[1]):
                if dm == tval:
                    image[j, k] = ival
                    break
    return domain, image


# ---------- Making Effective Mass Plots for Two-Point Function ---------- #
# unsophisticated meff-plot


def func(meff, t, ratio):
    """ main function """
    Nt = 64.0
    tau = t - Nt/2.0
    func_ratio = np.cosh(meff*(tau+1))/np.cosh(meff*tau)
    answer = func_ratio - ratio
    error_message = "Bad time slice. C(t+1)/C(t) <= 1 for all t < half the Time Length"
    if ratio > 1:
        raise ValueError(error_message)
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
    psq *= (2*np.pi/Nx)**2
    return np.sqrt(psq + mpisq)


def bisect_meff(func, Dat, a, b):
    """ gives me effective mass plot as a function of time: brentq """
    jk_smp = Dat.shape[0]
    T = np.arange(Dat.shape[-1])
    ratio = np.roll(Dat, -1, axis=-1).mean(axis=0)/Dat.mean(axis=0)
    cv_arglist = [(t, ratio[t_iter]) for t_iter, t in enumerate(T)]
    t_return = []
    cv = []
    for t, cv_arg in zip(T, cv_arglist):
        try:
            cv.append(opt.brentq(func, a, b, args=cv_arg))
            t_return.append(t)
        except ValueError:
            # print "skipping t = {0}".format(t)
            pass
    assert len(t_return) > 0
    t_return = np.asarray(t_return)
    cv = np.asarray(cv)
    return t_return, cv


def bisect_meff_mean_var(func, Dat, a, b):
    """
    another attempt
    """
    jk_smp = Dat.shape[0]
    Dat = Dat.real
    Blocks = lstats.jk_blocks(Dat, jk_smp, axis=0)
    main_vals = bisect_meff(func, Dat, a, b)
    val_list = [main_vals]
    for dat_block in Blocks:
        val_list.append(bisect_meff(func, dat_block, a, b))
    trange, image_list = match_lists(val_list)
    cv = image_list[0]
    diff_blocks = [pow(cv - cv_blk, 2) for cv_blk in image_list[1:]]
    variance = ft.reduce(lambda x, y: x + y, diff_blocks)
    variance *= (jk_smp - 1.0)/jk_smp
    return trange, cv, np.sqrt(variance)


# Phase Reweighting
def norm_c2pt(data):
    norm = data*data.conj()
    return np.sqrt(norm)


def phrw_c2pt(data, to):
    norm = norm_c2pt(data)
    phrw_phase = np.roll(norm, -to, axis=1)
    new_data = np.empty(data.shape, data.dtype)
    for j in xrange(data.shape[0]):
        new_data[j] = data[j]/phrw_phase[j]
    return new_data
