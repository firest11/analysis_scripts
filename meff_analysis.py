import functools as ft
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import lat_stats as lstats


# Auxillary Functions
def match_indices(list1, list2, match=True):
    """
    Returns common indices between list1 and list2
    list1 and list2 must be the same length
    If 'match'=False, returns indices where list1 != list2
    """
    indices = []
    for j, (l1, l2) in enumerate(zip(list1, list2)):
        compare = l1[j]==l2[j]
        if compare == match:
            indices.append(j)
    return indices


def large_match_indices(lists, match=True):
    """
    Returns common (uncommon) indices between a list of lists
    """
    check_lists = [len(lst)==len(lists[0]) for lst in lists[1:]]
    assert all(check_lists)
    list1 = lists[0]
    indices_list = [match_indices(list1, lst, match=match) for lst in lists]
    func_table = {'True': np.intersect1d, 'False': np.union1d}
    match_index = ft.reduce(func_table[str(match)], indices_list)
    return match_index


# Making Effective Mass Plots for Two-Point Function
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


def newton_meff(func, Dat, guess, fprime=None, fprime2=None, **kwargs):
    """ gives me effective mass plot as function of time: newton """
    jk_smp = Dat.shape[0]
    T = np.arange(Dat.shape[-1])
    ratio = np.roll(Dat, -1, axis=-1).mean(axis=0)/Dat.mean(axis=0)
    # cv == central value
    cv_arglist = [(t, ratio[t_iter]) for t_iter, t in enumerate(T)]
    t_return = []
    cv = []
    for t, cv_arg in zip(T, cv_arglist):
        try:
            cv.append(opt.newton(func, guess, args=cv_arg,
                                 fprime=fprime, fprime2=fprime2))
            t_return.append(t)
        except ValueError:
            print "skipping t = {0}".format(t)
    assert len(t_return) > 0
    t_return = np.asarray(t_return)
    cv = np.asarray(cv)
    return t_return, cv


def newton_meff_mean_var(func, Dat, guess, fprime=None, fprime2=None, **kwargs):
    """ 
    gives me effective mass plot as function of time with jackknife error: newton
    """
    jk_smp = Dat.shape[0]
    T = np.arange(Dat.shape[-1])
    Blocks = lstats.jk_blocks(Dat, jk_smp, axis=0)
    trange, cv = newton_meff(func, Dat, guess, fprime, fprime2, **kwargs)
    t_blocks, cv_blocks = [], []
    for blk in Blocks:
        tblk, cvblk = newton_meff(func, Dat, guess, fprime, fprime2, **kwargs)
        t_blocks.append(tblk)
        cv_blocks.append(cvblk)
    t_list_set = [trange]
    for tblk in t_blocks:
        t_list_set.append(tblk)
    match_index = large_match_indices(t_list_set) # match == True, take union
    trange = trange[match_index]
    cv = cv[match_index]
    for j, (tblk, cvblk) in enumerate(zip(t_blocks, cv_blocks)):
        t_blocks[j] = tblk[match_index]
        cv_blocks[j] = cvblk[match_index]
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
    ratio = np.roll(Dat, -1, axis=-1).mean(axis=0)/Dat.mean(axis=0)
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
