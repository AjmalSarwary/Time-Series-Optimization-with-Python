import pandas as pd
import numpy as np

def drawdown(return_series: pd.Series):
	"""Takes a time series of asset returns, 
		returns a DataFrame with columns for the 
		wealth index, 
		the previous peaks, and
		the percentage drawdown
	"""
    # how well can $1000 initial amt perform with the returns series cumulatively over time
	wealth_index = 1000*(1+return_series).cumprod()
    # for each period record the highest recent peak in wealth accumulation
	previous_peaks = wealth_index.cummax()
    # percentage of wealth lost, recorded over time/row-wise
	drawdowns = (wealth_index - previous_peaks)/previous_peaks
    
	return pd.DataFrame({"Wealth": wealth_index,
						"Previous Peak":previous_peaks,
						"Drawdown":drawdowns})


def semideviation(r):
    """
        A downside risk measure;
        Returns the semideviation aka negative semideviation of r since only deviations to the left of the mean are of a concern
        equivalent to: 'returns[returns<0].std(ddof=0)'
        r must be a Series or a DataFrame
        OR:
        Semideviation is the volatility of the sub-sample of below-average(belwo-zero) returns
    """
    # create mask/filter
    is_negative = r < 0
    # std for filtered part
    return r[is_negative].std(ddof=0)
    

def skewness(r):
    """
        Alternative to scipy.stats.skew()
        Computes the skewness of the supplied Series of DataFrame
        Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population std, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    
    return exp/sigma_r**3


def kurtosis(r):
    """
        Alternative to scipy.stats.kurtosis()
        Computes the kurtosis of the supplied Series or DataFrame
        Returns a float or a Series
    """
    demeaned_r = r-r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

def var_historic(r, level=5):
    """
        Returns the historic Value at Risk at a specified level i.e. returns the number such that "level" percent of the returns fall below that number, and the (100 - level) percent are above 
        In other words: looking at the worst possible outcome(the percentile) after excluding the worst level% of the outcomes which is the worst possible outcome at a confidence level of (100 - level)
    """
    if isinstance(r, pd.DataFrame):
        #aggregate runs a function over individual columns, i.e. Series
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        # negative of negative just for communicative purposes
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be Series or DataFrame")
        
from scipy.stats import norm
def var_gaussian(r, level=5, modified=False):
    """
        Returns the Parametric Gaussian VaR of a Series or DataFrame
        If "modified" is True, then the modified VaR is returned, using the Cornish-Fisher modification
    """
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    # compute z score for Cornish-Fisher
    if modified:
        s = skewness(r)
        k = kurtosis(r)
        # adjust z for Cornish-Fisher modification
        z = (z +
                (z**2 -1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
            
    return -(r.mean() + z*r.std(ddof=0))
    

def cvar_historic(r, level=5):
    """
        Computes the Conditional VaR of Series or DataFrame
        CVar: when the worst case happens, i.e. returns fall below the Var at level=level, the avg losses are CVar
    """
    if isinstance(r, pd.Series):
        # define mask
        is_beyond = r <= -var_historic(r, level=level)
        # apply mask on r
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be Series or DataFrame")


