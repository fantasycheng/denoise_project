import numpy as np
import pandas as pd
# import plotly.express as px
import plotly.graph_objects as go
import plotly.express as px

from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize

def mpPDF(var, q, pts):

    eMin = var * (1 - (1 / q) ** .5) ** 2
    eMax = var * (1 + (1 / q) ** .5) ** 2

    eVal = np.linspace(eMin, eMax, pts)
    if len(eVal.shape) == 2:
        eVal = eVal.squeeze()

    pdf = q / (2 * np.pi * var * eVal) * ((eMax - eVal) * (eVal - eMin)) ** .5
    pdf = pd.Series(pdf, index=eVal)

    return pdf

def getPCA(matrix):
    eVal, eVec = np.linalg.eigh(matrix)
    idx = eVal.argsort()[::-1]

    eVal = np.diagflat(eVal[idx])
    eVec = eVec[:, idx]

    return eVal, eVec

def fitKDE(obs, bWidth=.25, kernel='gaussian', x=None):
    if len(obs.shape) == 1:
        obs = obs.reshape(-1, 1)
    
    kde = KernelDensity(kernel=kernel, bandwidth=bWidth).fit(obs)

    if x is None:
        x = np.unique(obs).reshape(-1, 1)
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    
    log_prob = kde.score_samples(x)
    pdf = pd.Series(np.exp(log_prob), index=x.flatten())

    return pdf

def get_random_cov(n_cols, n_facts):
    w = np.random.normal(size=(n_cols, n_facts))

    cov = np.dot(w, w.T)
    cov += np.diag(np.random.uniform(size=n_cols))

    return cov

def cov2corr(cov):
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)

    corr[corr < -1] = -1
    corr[corr > 1] = 1

    return corr

def err_pdfs(var, e_val, q, bWidth, pts=1000):
    pdf0 = mpPDF(var, q, pts)
    pdf1 = fitKDE(e_val, bWidth, x=pdf0.index.values)
    sse = np.sum((pdf1 - pdf0) ** 2)

    return sse

def find_max_eval(e_val, q, bWidth):
    out = minimize(lambda *x: err_pdfs(*x), 0.5, args=(e_val, q, bWidth), bounds=((1e-5, 1-1e-5), ))
    if out['success']:
        var = out['x'][0]
    else:
        var = 1
    
    e_max = var * (1 + (1 / q) ** 0.5) ** 2
    return e_max, var

def denoise_corr(e_val, e_vec, n_facts):
    e_val_ = np.diag(e_val).copy()
    e_val_[n_facts : ] = e_val_[n_facts : ].mean()
    e_val_ = np.diag(e_val_)

    corr = np.dot(e_vec, e_val_).dot(e_vec.T)
    corr = cov2corr(corr)

    return corr

if __name__ == '__main__':
    # ---------
    # 拟合MP分布
    # ---------

    # x = np.random.normal(size=(10000, 1000))
    # eVal, eVec = getPCA(np.corrcoef(x, rowvar=0))

    # pdf0 = mpPDF(1, q=x.shape[0]/x.shape[1], pts=1000)
    # pdf1 = fitKDE(np.diag(eVal), bWidth=.01)

    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=pdf0.index, y=pdf0, mode='lines', name='Marcenko-Pastur'))
    # fig.add_trace(go.Scatter(x=pdf1.index, y=pdf1, mode='markers', name='Empirical:KDE'))
    # fig.show()

    # ---------
    # 拟合MP分布
    # ---------

    alpha = 0.998
    n_cols = 335
    n_facts = 14
    q = 11
    bWidth = 0.01
    nbinsx = n_cols
    # nbinsx = int(n_cols / 10)

    cov = np.cov(np.random.normal(size=(n_cols * q, n_cols)), rowvar=0)
    cov = alpha * cov + (1 - alpha) * get_random_cov(n_cols, n_facts)
    corr = cov2corr(cov)

    e_val, e_vec = getPCA(corr)
    e_max, var = find_max_eval(np.diag(e_val), q, bWidth=bWidth)
    # n_pred_facts = e_val.shape[0] - np.diag(e_val)[::-1].searchsorted(e_max)
    n_pred_facts = (np.diag(e_val) > e_max).sum()
    print(n_facts, n_pred_facts, var)

    # mp_pdf = mpPDF(var, q, pts=1000)
    # emp_pdf = fitKDE(np.diag(e_val), bWidth, x=mp_pdf.index.values)

    # ## Fitting the Marcenko-Pastur PDF on a noisy covariance matrix

    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=mp_pdf.index, y=mp_pdf, mode='lines', name='Marcenko-Pastur'))
    # # fig.add_trace(go.Scatter(x=emp_pdf.index, y=emp_pdf, mode='markers', name='Empirical Dist'))
    # fig.add_trace(go.Histogram(x=np.diag(e_val), histnorm='probability density', nbinsx=nbinsx, name='Empirical Dist'))
    # fig.show()

    # dn_corr = denoise_corr(e_val, e_vec, n_pred_facts)
    # dn_eval, dn_evec = getPCA(dn_corr)

    # df = pd.DataFrame({'origin_eval': np.diag(e_val), 'denoisied_eval': np.diag(dn_eval)})
    # fig = px.ecdf(df, log_y=['origin_eval', 'denoisied_eval'], ecdfmode='reversed', ecdfnorm=None, orientation='h')
    # # fig = px.ecdf(df, log_y=['origin_eval', 'denoisied_eval'], ecdfmode='reversed', ecdfnorm=None, orientation='h', marginal='histogram')
    # fig.show()



