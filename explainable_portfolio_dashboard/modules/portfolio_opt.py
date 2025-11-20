import numpy as np

def mean_variance_opt(returns, target_return=None):
    """
    returns: DataFrame or 2D np array of asset returns (rows = dates, cols = assets)
    target_return: optional desired portfolio return (annualized unit same as returns)
    Returns: weights np.array
    Fallback: if scipy not available, return equal weights.
    """
    X = returns
    try:
        import pandas as pd
        if hasattr(X, "values"):
            mu = X.mean().values
            Sigma = X.cov().values
        else:
            mu = np.mean(X, axis=0)
            Sigma = np.cov(X, rowvar=False)
        n = len(mu)
    except Exception:
        # If something unexpected, fallback to equal weights
        n = X.shape[1] if hasattr(X, "shape") else 1
        return np.ones(n)/n

    # simple mean-variance using scipy minimize (if available)
    try:
        from scipy.optimize import minimize
        def port_var(w): return w.T @ Sigma @ w
        cons = ({'type':'eq','fun': lambda w: np.sum(w)-1.0},)
        if target_return is not None:
            cons = cons + ({'type':'eq','fun': lambda w: w.dot(mu) - target_return},)
        bounds = tuple((0,1) for _ in range(len(mu)))
        x0 = np.ones(len(mu))/len(mu)
        res = minimize(port_var, x0, bounds=bounds, constraints=cons)
        if res.success:
            return np.maximum(res.x, 0) / np.sum(np.maximum(res.x, 0))
        else:
            # fallback equal weights
            return np.ones(len(mu))/len(mu)
    except Exception:
        # SciPy not available: quick approximate weights proportional to positive mu
        pos = np.clip(mu, a_min=0, a_max=None)
        s = pos.sum()
        if s <= 0:
            return np.ones(len(mu))/len(mu)
        return pos / s
