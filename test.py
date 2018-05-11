import numpy as np
import lat_stats as lstat


# Double Check with Someone Else's Code
def jackknife(x, func):
    """Jackknife estimate of the estimator func"""
    n = len(x)
    idx = np.arange(n)
    return np.sum(func(x[idx!=i]) for i in range(n))/float(n)


def jackknife_var(x, func):
    """Jackknife estiamte of the variance of the estimator func."""
    n = len(x)
    idx = np.arange(n)
    j_est = jackknife(x, func)
    return (n-1)/(n + 0.0) * np.sum((func(x[idx!=i]) - j_est)**2.0
                                    for i in range(n))


def jackknife_mean_err(x, func):
    """ Just does the both above """
    estimate = jackknife(x, func)
    variance = jackknife_var(x, func)
    return estimate, np.sqrt(variance)

# Determine the mean and variance of three normal distributions
Norm1 = np.random.normal(0, 1.0, 10)
Norm2 = np.random.normal(0, 1.0, 100)
Norm3 = np.random.normal(0, 1.0, 1000)

# Number of jk_smps
bins1 = [2, 5, 10]
bins2 = [2, 5, 10, 25, 50, 100]
bins3 = [2, 5, 10, 25, 50, 100, 250, 500, 1000]

# Jk_blocks
blocks1 = [lstat.jk_blocks(Norm1, b_1) for b_1 in bins1]
blocks2 = [lstat.jk_blocks(Norm2, b_2) for b_2 in bins2]
blocks3 = [lstat.jk_blocks(Norm3, b_3) for b_3 in bins3]

# Now getting mean and sigma
mean1, std1 = Norm1.mean(), Norm1.std()
mean2, std2 = Norm2.mean(), Norm2.std()
mean3, std3 = Norm3.mean(), Norm3.std()

regdat1 = (mean1, std1)
regdat2 = (mean2, std2)
regdat3 = (mean3, std3)

mydat1 = lstat.calc_mean_var(Norm1)
mydat2 = lstat.calc_mean_var(Norm2)
mydat3 = lstat.calc_mean_var(Norm3)

jkdat1 = [lstat.calc_jk_mean_var(Norm1, block1) for block1 in blocks1]
jkdat2 = [lstat.calc_jk_mean_var(Norm2, block2) for block2 in blocks2]
jkdat3 = [lstat.calc_jk_mean_var(Norm3, block3) for block3 in blocks3]

jk_imp_dat1 = [lstat.calc_bias_corrected_jk_mean_var(Norm1, block1)
               for block1 in blocks1]
jk_imp_dat2 = [lstat.calc_bias_corrected_jk_mean_var(Norm2, block2)
               for block2 in blocks2]
jk_imp_dat3 = [lstat.calc_bias_corrected_jk_mean_var(Norm3, block3)
               for block3 in blocks3]

check_test1 = jackknife_mean_err(Norm1, np.mean)
check_test2 = jackknife_mean_err(Norm2, np.mean)
check_test3 = jackknife_mean_err(Norm3, np.mean)
