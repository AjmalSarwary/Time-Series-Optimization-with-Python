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

def annualize_rets(r, periods_per_year):
    """
        Annualizes set of returns
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1

def annualize_vol(r, periods_per_year):
    """
    Annualizes the volatility of a set of returns 
    """
    # scale the std by the number of the sqrt of the periods
    return r.std()*(periods_per_year**0.5)

def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    #convert the annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol

def portfolio_return(weights, returns):
    """
    Weights -> Retruns
    """
    # matrix multipication of wights with returns
    return weights.T @ returns
    
    
def portfolio_vol(weights, covmat):
    """
    Weights -> Volatility
    """
    return (weights.T  @ covmat @ weights)**0.5


def plot_ef2(n_points, er, cov, style):
    """
    Plots the 2-asset efficient frontier
    """
    if er.shape[0] !=2:
        raise ValueError("plot_ef2 can only plot 2-asset frontiers")
    weights = [np.array([w,1-w]) for w in np.linspace(0,1,n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({"Returns": rets, "Vols": vols})
    return ef.plot.line(x="Vols", y="Returns", style=style)
  
from scipy.optimize import minimize  
def minimize_vol(target_return, er, cov):
    """
    target_ret -> w
    takes in a desired target return, expeced returns Series, and a covariance matrix
    returns the optimal weights for the target return with lowest possible volatility
    """
    # number of portfolios
    n = er.shape[0]
    # initial allocation across portfolios
    init_guess = np.repeat(1/n, n)
    #constains:
    # n tuples trick to make n-tuples for the bounds of the allocations
    # bounds for each weight must be 0=< bound =<1
    bounds = ((0.0, 1.0),)*n
    # define target return as an equality constraint
    # target return - rk.portfolio_return() = 0
    return_is_target = {
        'type': 'eq',
        'args': (er,),
        'fun': lambda weights, er: target_return - portfolio_return(weights,er)
    }
    # weights must sum to 1, i.e. 'sum(weights) -1 = 0
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
   
    # objective function: quadratic optimizer SLSQP
    results = minimize(portfolio_vol, 
                       init_guess, 
                       args=(cov,), 
                       method="SLSQP",
                       options={'disp':False},
              constraints=(return_is_target, 
                           weights_sum_to_1),
                           bounds=bounds
    )
    # return only weights
    return results.x
    
    

def get_ffme_returns():
  """
  Load the Fama-French Dataset for the retruns of the Top and Bottom Deciles by MarketCap
  ffme_returns: fama-french market equity based returns
  """
  file_path = '/content/invest_ml/data/Portfolios_Formed_on_ME_monthly_EW.csv'
  rets = pd.read_csv(file_path, header=0, index_col=0, na_values=99.99)
  rets = rets[['Lo 10','Hi 10']]
  rets.columns = ['SmallCap', 'LargeCap']
  rets = rets/100
  rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period('M')
  return rets

def get_hfi_returns():
  """
  Load and format the EDHEC Hedge Fund Index Returns
  Dataset contains various kinds of Hedge Fund Strategies (Hedge Funds Indices) produced by EDHEC from 1997 onward
  """
  file_path = '/content/invest_ml/data/edhec-hedgefundindices.csv'
  hfi = pd.read_csv(file_path, header=0, index_col=0, parse_dates=True)
  hfi = hfi/100
  hfi.index = hfi.index.to_period('M')
  return hfi
  

def get_ind_returns():
    """
    Load and format ind30_m_vw_rets represents monthly returns of 30 different industry portfolios fom 1926 onward
    """
    file_path = '/content/invest_ml/data/ind30_m_vw_rets.csv'
    ind = pd.read_csv(file_path, header=0, index_col=0, parse_dates=True)
    ind = ind/100
    ind.index = pd.to_datetime(ind.index, format='%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind


def optimal_weights(n_points, er, cov):
  """
  -> list of weights to run the optimizer to minimize the vol
  """
  target_rs = np.linspace(er.min(), er.max(), n_points)
  weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
  return weights

def msr(riskfree_rate,er, cov):
    """
    Riskfree rate + ER + COV -> w
    Return the weights of the portfolio that gives the maximum sharpe ratio given the riskfree rate and expected returns and a covariance matrix
 
    """
    # number of portfolios
    n = er.shape[0]
    # initial allocation across portfolios
    init_guess = np.repeat(1/n, n)
    #constains:
    # n tuples trick to make n-tuples for the bounds of the allocations
    # bounds for each weight must be 0=< bound =<1
    bounds = ((0.0, 1.0),)*n

   
    # weights must sum to 1, i.e. 'sum(weights) -1 = 0
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    
    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
      """
      Returns the negative of the sharpe ratio
      """
      r = portfolio_return(weights, er)
      vol = portfolio_vol(weights, cov)
      return -(r - riskfree_rate)/vol

   
    # objective function: quadratic optimizer SLSQP
    results = minimize(neg_sharpe_ratio, 
                       init_guess, 
                       args=(riskfree_rate,er,cov,), 
                       method="SLSQP",
                       options={'disp':False},
              constraints=(weights_sum_to_1),
                           bounds=bounds
    )
    # return only weights
    return results.x


def plot_ef(n_points, er, cov, show_cml=True, style='.-', riskfree_rate=0):
  """
  Plots the multi-asset efficient frontier
  """
  weights = optimal_weights(n_points, er, cov)
  rets = [portfolio_return(w,er) for w in weights]
  vols = [portfolio_vol(w,cov) for w in weights]
  ef = pd.DataFrame({
      "Returns": rets,
      "Volatility": vols
  })
  ax = ef.plot.line(x="Volatility", y="Returns", style=style)
  if show_cml:
        ax.set_xlim(left=0)
    rf = 0.1
    w_msr = msr(rf, er, cov)
    r_msr = portfolio_return(w_msr, er)
    vol_msr = portfolio_vol(w_msr, cov)
    # Add CML
    cml_x = [0, vol_msr]
    cml_y = [riskfree_rate, r_msr]
    ax.plot(cml_x, cml_y, color= "chartreuse", marker="o", linestyle="dashed", markersize=12, linewidth=2)
    return ax