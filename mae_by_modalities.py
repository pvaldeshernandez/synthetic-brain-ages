import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from arch.bootstrap import IIDBootstrap


# defines the function to calculate the median absolute error
def getAB(T, avar, bavar):
    indIDu = pd.factorize(T["ID"])[0]
    a = T.groupby(indIDu)[avar].apply(list)
    b = T.groupby(indIDu)[bavar].apply(list)
    n = a.apply(len)
    aa = np.array(a.apply(lambda x: x + [np.nan] * (n.max() - len(x))).tolist())
    bb = np.array(b.apply(lambda x: x + [np.nan] * (n.max() - len(x))).tolist())
    return aa, bb


def calculateR2(A, B):
    n = np.sum(~np.isnan(A), axis=1)
    shape = A.shape
    subx = np.arange(shape[0])
    suby = np.random.randint(n)
    ind = np.ravel_multi_index((subx, suby), shape)
    as_ = A.ravel()[ind]
    bs = B.ravel()[ind]
    reg = LinearRegression().fit(as_, bs)
    r2 = r2_score(bs, reg.predict(as_))
    return r2


def calculateR(A, B):
    n = np.sum(~np.isnan(A), axis=1)
    shape = A.shape
    subx = np.arange(shape[0])
    suby = np.random.randint(n)
    ind = np.ravel_multi_index((subx, suby), shape)
    as_ = A.ravel()[ind]
    bs = B.ravel()[ind]
    r = np.corrcoef(as_, bs, rowvar=False)
    return r


def calculateMAE(A, B):
    n = np.sum(~np.isnan(A), axis=1)
    shape = A.shape
    subx = np.arange(shape[0])
    suby = np.random.randint(n)
    ind = np.ravel_multi_index((subx, suby), shape)
    as_ = A.ravel()[ind]
    bs = B.ravel()[ind]
    mae = np.nanmean(np.abs(as_ - bs))
    return mae


def bootstrap(nboots: int, func: callable, *args):
    bs = IIDBootstrap(*args)
    ci_lower_upper_mean_samples = bs.conf_int(
        func, reps=nboots, method="percentile", size=0.05, tail="two"
    )

    ci_lower_upper_mean_samples[3] *= nboots
    return ci_lower_upper_mean_samples[:3]


def BAstats(T, avar, bavar, groupvar, nboots, r):
    if groupvar is None:
        Ng = 1
    else:
        groups = T[groupvar].cat.categories
        Ng = len(groups)

    if r > 0:
        MAECIr = np.zeros((Ng, 2))
        RCIr = np.zeros((Ng, 2))
        R2CIr = np.zeros((Ng, 2))
        MAEr = np.zeros(Ng)
        Rr = np.zeros(Ng)
        R2r = np.zeros(Ng)
    else:
        MAECIr = []
        RCIr = []
        R2CIr = []
        MAEr = []
        Rr = []
        R2r = []

    MAECIm = np.zeros((Ng, 2))
    RCIm = np.zeros((Ng, 2))
    R2CIm = np.zeros((Ng, 2))
    MAEm = np.zeros(Ng)
    Rm = np.zeros(Ng)
    R2m = np.zeros(Ng)

    for g in range(Ng):
        Tg = T[T[groupvar] == groups[g]]
        A, B = getAB(Tg, avar, bavar)

        if r > 0:
            MAECIr[g], MAEsr = bootstrap(r * nboots, calculateMAE, A, B)
            MAEr[g] = np.mean(MAEsr)

        MAECIm[g], MAEsm = bootstrap(
            nboots,
            lambda x: np.nanmean(x),
            np.abs(np.nanmedian(A, axis=1) - np.nanmedian(B, axis=1)),
        )
        MAEm[g] = np.mean(MAEsm)

        if r > 0:
            RCIr[g], Rsr = bootstrap(r * nboots, calculateR, A, B)
            Rr[g] = np.mean(Rsr)

        RCIm[g], Rsm = bootstrap(
            nboots,
            lambda x: np.corrcoef(x[0], x[1])[0][1],
            (np.nanmedian(A, axis=1), np.nanmedian(B, axis=1)),
        )
        Rm[g] = np.mean(Rsm)

        if r > 0:
            R2CIr[g], R2sr = bootstrap(r * nboots, calculateR2, A, B)
            R2r[g] = np.mean(R2sr)

        R2CIm[g], R2sm = bootstrap(
            nboots, calculateR2, (np.nanmedian(A, axis=1), np.nanmedian(B, axis=1))
        )
        R2m[g] = np.mean(R2sm)

        if r > 0:
            MAEr = [f"{x:.2f}" for x in MAEr]
            MAECIr = [f"[{x:.2f}, {y:.2f}]" for x, y in MAECIr]

        MAEm = [f"{x:.2f}" for x in MAEm]
        MAECIm = [f"[{x:.2f}, {y:.2f}]" for x, y in MAECIm]

        if r > 0:
            Rr = [f"{x:.2f}" for x in Rr]
            RCIr = [f"[{x:.2f}, {y:.2f}]" for x, y in RCIr]

        Rm = [f"{x:.2f}" for x in Rm]
        RCIm = [f"[{x:.2f}, {y:.2f}]" for x, y in RCIm]

        if r > 0:
            R2r = [f"{x:.2f}" for x in R2r]
            R2CIr = [f"[{x:.2f}, {y:.2f}]" for x, y in R2CIr]

        R2m = [f"{x:.2f}" for x in R2m]
        R2CIm = [f"[{x:.2f}, {y:.2f}]" for x, y in R2CIm]

        pd_mae = pd.DataFrame(
            {
                "groups": groups,
                "MAEr": MAEr,
                "MAECIr": MAECIr,
                "MAEm": MAEm,
                "MAECIm": MAECIm,
            }
        )
        pd_corr = pd.DataFrame({"groups": groups, "Rr": Rr, "RCIr": RCIr, "Rm": Rm, "RCIm": RCIm})
        pd_Rsquared = pd.DataFrame(
            {"groups": groups, "R2r": R2r, "R2CIr": R2CIr, "R2m": R2m, "R2CIm": R2CIm}
        )
        pd_all = pd.DataFrame(
            {
                "groups": groups,
                "MAEr": MAEr,
                "MAECIr": MAECIr,
                "MAEm": MAEm,
                "MAECIm": MAECIm,
                "Rr": Rr,
                "RCIr": RCIr,
                "Rm": Rm,
                "RCIm": RCIm,
                "R2r": R2r,
                "R2CIr": R2CIr,
                "R2m": R2m,
                "R2CIm": R2CIm,
            }
        )

    return pd_mae, pd_corr, pd_Rsquared, pd_all
