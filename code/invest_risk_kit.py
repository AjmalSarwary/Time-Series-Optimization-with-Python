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
        Returns the semideviation aka negative semideviation of r 
        r must be a Series or a DataFrame
    """
    is_negative = r < 0
    return r[is_negative].std(ddof=0)
    

def skewness(r):
    """
        Alternative to scipy.stats.skew()
        Computes the skewness of the supplied Series of DataFrame
        Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population std, so set dof=0
    sigma_r = r.str(ddof=0)
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


