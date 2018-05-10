import numpy as np
import scipy.optimize as opt


L = [48, 48, 48, 64]
print "Lattice Size = {0}".format(L)

def polyfit(pars, x):
    """
    generic polynomial function to the order of pars
    polyfit(x) = sum_j pars[j]*x**j
    """
    if type(pars) is not list:
        pars = [pars]
    val = 0
    for j, par in enumerate(pars):
        val += par*(x**j)
    return val


def exp_fit(pars, x):
    """
    generic sum of exponentials
    exp_fit(x) = sum_(j even) pars[j]*exp(-pars[j+1]*x)
    """
    if len(pars) % 2 != 0:
        raise "Need at least two parameters"
    val = 0
    for j in xrange(len(pars), 2):
        val += pars[j]*np.exp(-pars[j+1]*x)
    return val


def exp_fit_bnds(pars, x, tag):
    """
    Sum of exponentials for either mesons
    or baryons. Includes 'wrap-around' effect
    that come from boundary conditions on the 
    finite lattice (periodic boundary condition).
    """
    if len(pars) % 2 != 0:
        raise "Need at least two parameters"
    if tag not in ['meson', 'baryon']:
        raise "Tag must be 'meson' or 'baryon'"
    sgn = {'meson': 1.0, 'baryon': -1.0}
    val = 0
    for j in xrange(len(pars), 2):
        exps = np.exp(-pars[j+1]*x) + sgn[tag]*np.exp(-pars[j+1]*(L[-1]-x))
        val += pars[j]*exps
    return val
