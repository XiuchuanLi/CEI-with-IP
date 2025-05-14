import numpy as np
from scipy.stats import pearsonr
from kerpy.GaussianKernel import GaussianKernel
from independence_testing.HSICSpectralTestObject import HSICSpectralTestObject


def correlation(x, y, alpha=0.05):
    rho, p_value = pearsonr(x, y)
    if p_value < alpha:
        return True, p_value
    else:
        return False, p_value
    

def independence(x, y, alpha=0.05):
    lens = len(x)
    x=x.reshape(lens,1)
    y=y.reshape(lens,1)
    kernelY = GaussianKernel(float(1.0))
    kernelX=GaussianKernel(float(1.0))
    num_samples = lens

    myspectralobject = HSICSpectralTestObject(num_samples, kernelX=kernelX, kernelY=kernelY,
                                          kernelX_use_median=False, kernelY_use_median=False,
                                          rff=True, num_rfx=30, num_rfy=30, num_nullsims=1000)
    p_value = myspectralobject.compute_pvalue(x, y)

    if p_value > alpha:
        return True, p_value
    else:
        return False, p_value


# unbiased estimator of cum(x,x,x,x)
def cum4(x):
    n = len(x)
    cumulant = (n+2) / (n * (n-1)) * np.sum(x ** 4) - 3 / (n * (n-1)) * (np.sum(x ** 2) ** 2)
    return cumulant


# unbiased estimator of cum(x,x,x,y)
def cum31(x,y):
    n = len(x)
    cumulant = np.mean(x**3 * y) - 3 / (n**2 - n) * (np.sum(x**2) * np.sum(x * y) - np.sum(x**3 * y))
    return cumulant


# unbiased estimator of cum(x,x,y,y)
def cum22(x,y):
    n = len(x)
    cumulant = np.mean(x**2 * y**2) - 1 / (n**2 - n) * (np.sum(x**2) * np.sum(y**2) - np.sum(x**2 * y**2)) - 2 / (n**2 - n) * (np.sum(x*y)**2 - np.sum(x**2 * y**2))
    return cumulant


def candidates(x, y):
    a = cum31(x,y) ** 2 - cum22(x,y) * cum4(x)
    b = cum31(y,x) * cum4(x) - cum31(x,y) * cum22(x,y)
    c = cum22(x,y) ** 2 - cum31(x,y) * cum31(y,x)
    delta = b*b - 4*a*c
    root1 = (-b + np.sqrt(np.abs(delta))) / (2*a)
    root2 = (-b - np.sqrt(np.abs(delta))) / (2*a)
    return root1, root2


def ind_constraint(x1,x2,x3):
    flag1 = correlation(x1,x2)[0]
    flag2 = correlation(x1,x3)[0]
    flag3, p_value = independence(np.cov(x2,x3)[0,1]*x1 - np.cov(x1,x3)[0,1]*x2, x3)
    return flag1 and flag2 and flag3, p_value


def performance(ndarr, ratio = 0.1):
    ndarr_sorted = np.sort(ndarr)
    ndarr_trimed = ndarr_sorted[int(len(ndarr)*ratio):int(len(ndarr)*(1-ratio))]
    return ndarr_trimed.mean(), ndarr_trimed.std()