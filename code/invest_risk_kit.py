"""
This script is a comprehensive suite of custom functions for portfolio optimization, financial risk, and market analysis. Here is an overview of the functionality each function provides:

- `drawdown`: Calculates the drawdowns, representing declines from a historical peak in a wealth index.
- `semideviation`: Measures the downside risk of returns, focusing on negative variations.
- `skewness`: Assesses the asymmetry of the return distribution of an asset.
- `kurtosis`: Evaluates the 'tailedness' of the return distribution and identifies fat-tailed or thin-tailed distributions.
- `var_historic`: Computes the historical Value at Risk at a specified confidence level, indicating the maximum expected loss.
- `var_gaussian`: Estimates the Gaussian Value at Risk, with an option for the Cornish-Fisher adjustment.
- `cvar_historic`: Calculates the Conditional Value at Risk, the average loss exceeding the VaR level.
- `annualize_rets`: Converts shorter-period returns into annual returns.
- `annualize_vol`: Converts shorter-period volatility into annualized volatility.
- `sharpe_ratio`: Computes the annualized Sharpe ratio of a set of returns, providing a risk-adjusted measure of return.
- `portfolio_return`: Calculates the overall return of a portfolio based on individual asset returns and weights.
- `portfolio_vol`: Determines the overall volatility of a portfolio given its asset weights and covariance matrix.
- `plot_ef2`: Plots the 2-asset efficient frontier, visualizing the trade-off between risk and return.
- `minimize_vol`: Finds the portfolio weights that minimize volatility for a given target return.
- `get_ffme_returns`: Loads the Fama-French Dataset for analyzing market equity returns segmented by size deciles.
- `get_hfi_returns`: Retrieves the EDHEC Hedge Fund Index Returns, offering insights into different hedge fund strategies.
- `get_ind_returns`: Provides monthly returns of various industry portfolios, facilitating an industry-wide analysis.
- `optimal_weights`: Calculates the weights that minimize the portfolio's volatility for a range of expected returns.
- `msr`: Identifies the Maximum Sharpe Ratio Portfolio given the risk-free rate and the expected returns and covariance of the assets.
- `neg_sharpe_ratio`: Computes the negative Sharpe ratio, often used as an objective function for optimization.
- `gmv`: Calculates the weights of the Global Minimum Volatility (GMV) portfolio based on the covariance matrix. The GMV portfolio aims to achieve the lowest possible volatility, focusing solely on risk minimization without considering expected returns.
- `plot_ef`: Plots the efficient frontier for a multi-asset portfolio, illustrating the trade-off between risk and return. The function also optionally shows the Capital Market Line (CML), Equal Weight (EW) portfolio, and Global Minimum Volatility (GMV) portfolio, providing a comprehensive view of different investment strategies and their risk-return profiles.
"""


import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize 

def drawdown(return_series: pd.Series):
    """
    Calculates the drawdown, which measures the decline from a historical peak in some variable (typically the cumulative return of an investment).

    Parameters:
    return_series (pd.Series): Time series of asset returns.

    Returns:
    pd.DataFrame: DataFrame containing columns for Wealth Index, Previous Peaks, and Drawdowns.
    """
    # Calculates the wealth index by cumulatively multiplying the returns (assuming a starting value of 1000).
    wealth_index = 1000 * (1 + return_series).cumprod()

    # Tracks the maximum value reached by the wealth index up to each point in time.
    previous_peaks = wealth_index.cummax()

    # Calculates the drawdown, which is the percentage loss from the previous peak.
    drawdowns = (wealth_index - previous_peaks) / previous_peaks

    return pd.DataFrame({"Wealth": wealth_index, "Previous Peak": previous_peaks, "Drawdown": drawdowns})



def semideviation(r):
    """
    Calculates the semideviation (downside risk) of returns. Semideviation only considers the cases where the returns are less than zero.

    Parameters:
    r (pd.Series or pd.DataFrame): Asset returns.

    Returns:
    float: Semideviation of the returns.
    """
    # Filter the returns to only include negative returns.
    is_negative = r < 0

    # Calculate the standard deviation of the filtered negative returns.
    return r[is_negative].std(ddof=0)

    

def skewness(r):
    """
    Computes the skewness of a set of returns. Skewness measures the asymmetry of the distribution of returns.

    Parameters:
    r (pd.Series or pd.DataFrame): Asset returns.

    Returns:
    float or pd.Series: Skewness of the returns.
    """
    # Subtract the mean from returns to get the demeaned returns.
    demeaned_r = r - r.mean()

    # Calculate the standard deviation of returns.
    sigma_r = r.std(ddof=0)

    # Calculate the mean of the cubed demeaned returns.
    exp = (demeaned_r**3).mean()

    # Divide the mean by the cube of the standard deviation to get skewness.
    return exp / sigma_r**3



def kurtosis(r):
    """
    Computes the kurtosis of a set of returns. Kurtosis measures the "tailedness" of the distribution of returns.

    Parameters:
    r (pd.Series or pd.DataFrame): Asset returns.

    Returns:
    float or pd.Series: Kurtosis of the returns.
    """
    # Subtract the mean from returns to get the demeaned returns.
    demeaned_r = r - r.mean()

    # Calculate the standard deviation of returns.
    sigma_r = r.std(ddof=0)

    # Calculate the mean of the quartic (fourth power) of the demeaned returns.
    exp = (demeaned_r**4).mean()

    # Divide the mean by the fourth power of the standard deviation to get kurtosis.
    return exp / sigma_r**4


def var_historic(r, level=5):
    """
    Calculates the historical Value at Risk (VaR) at a specified confidence level. VaR measures the maximum expected loss over a specified time frame.

    Parameters:
    r (pd.Series or pd.DataFrame): Asset returns.
    level (int): Confidence level (default is 5).

    Returns:
    float: Historic VaR at the specified level.
    """
    # Check if the input is a DataFrame and apply the function to each column.
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)

    # If the input is a Series, calculate the percentile.
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)

    else:
        raise TypeError("Expected r to be Series or DataFrame")

        

def var_gaussian(r, level=5, modified=False):
    """
    Calculates the Gaussian Value at Risk (VaR) using the normal distribution or the modified VaR using the Cornish-Fisher expansion.

    Parameters:
    r (pd.Series or pd.DataFrame): Asset returns.
    level (int): Confidence level (default is 5).
    modified (bool): Whether to use the modified VaR.

    Returns:
    float: Gaussian or modified VaR at the specified level.
    """
    # Calculate the Z-score from the normal distribution for the specified confidence level.
    z = norm.ppf(level / 100)

    # Adjust the Z-score using the Cornish-Fisher expansion if 'modified' is True.
    if modified:
        s = skewness(r)
        k = kurtosis(r)
        z = (z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * (k - 3) / 24 - (2 * z**3 - 5 * z) * (s**2) / 36)

    # Calculate the Gaussian VaR.
    return -(r.mean() + z * r.std(ddof=0))

    

def cvar_historic(r, level=5):
    """
    Calculates the Conditional Value at Risk (CVaR), also known as the Expected Shortfall. CVaR measures the average loss exceeding the VaR level.

    Parameters:
    r (pd.Series or pd.DataFrame): Asset returns.
    level (int): Confidence level (default is 5).

    Returns:
    float: CVaR at the specified level.
    """
    # If the input is a Series, calculate CVaR.
    if isinstance(r, pd.Series):
        # Determine which returns are beyond the historic VaR.
        is_beyond = r <= -var_historic(r, level=level)

        # Calculate the mean of the returns that are beyond the VaR.
        return -r[is_beyond].mean()

    # If the input is a DataFrame, apply the function to each column.
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)

    else:
        raise TypeError("Expected r to be Series or DataFrame")


def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns.
    
    Parameters:
    r (Series/DataFrame): Periodic returns of an asset or portfolio.
    periods_per_year (int): Number of periods in a year (e.g., 12 for monthly, 252 for daily).

    Returns:
    float: Annualized return of the asset or portfolio.
    """
    # Calculate the compounded growth rate by multiplying the returns.
    compounded_growth = (1 + r).prod()

    # Count the number of periods in the dataset.
    n_periods = r.shape[0]

    # Adjust the growth rate to an annual scale and subtract 1 to get the annualized return.
    return compounded_growth**(periods_per_year / n_periods) - 1

def annualize_vol(r, periods_per_year):
    """
    Annualizes the volatility of a set of returns.

    Parameters:
    r (Series/DataFrame): Periodic returns of an asset or portfolio.
    periods_per_year (int): Number of periods in a year (e.g., 12 for monthly, 252 for daily).

    Returns:
    float: Annualized volatility of the asset or portfolio.
    """
    # The annualized volatility is the standard deviation of returns, scaled by the square root of the number of periods in a year.
    return r.std() * (periods_per_year**0.5)

def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized Sharpe ratio of a set of returns.

    Parameters:
    r (Series/DataFrame): Periodic returns of an asset or portfolio.
    riskfree_rate (float): The risk-free rate per year.
    periods_per_year (int): Number of periods in a year.

    Returns:
    float: Annualized Sharpe ratio.
    """
    # Convert the annual risk-free rate to a periodic rate.
    rf_per_period = (1 + riskfree_rate)**(1 / periods_per_year) - 1

    # Calculate the excess returns by subtracting the risk-free rate from returns.
    excess_ret = r - rf_per_period

    # Annualize the excess returns and volatility.
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)

    # The Sharpe ratio is the annualized excess return divided by the annualized volatility.
    return ann_ex_ret / ann_vol


def portfolio_return(weights, returns):
    """
    Calculates the return of a portfolio based on asset weights and returns.

    Parameters:
    weights (array): Weights of different assets in the portfolio.
    returns (array): Returns of the individual assets.

    Returns:
    float: Total return of the portfolio.
    """
    # Multiply the weights by the returns and sum them up to get the portfolio return.
    # This is a matrix multiplication of weights (transpose) with the returns.
    return weights.T @ returns

    
    
def portfolio_vol(weights, covmat):
    """
    Calculates the volatility of a portfolio based on asset weights and covariance matrix.

    Parameters:
    weights (array): Weights of different assets in the portfolio.
    covmat (DataFrame/Matrix): Covariance matrix of the asset returns.

    Returns:
    float: Volatility of the portfolio.
    """
    # The portfolio volatility is calculated as the square root of the dot product of weights, covariance matrix, and weights.
    # It represents the standard deviation of portfolio returns.
    return (weights.T @ covmat @ weights)**0.5



def plot_ef2(n_points, er, cov, style):
    """
    Plots the efficient frontier for a portfolio of two assets.

    Parameters:
    n_points (int): Number of points to plot on the frontier.
    er (pd.Series): Expected returns for the two assets.
    cov (pd.DataFrame): Covariance matrix for the two assets.
    style (str): Style of the plot line.

    Returns:
    matplotlib.axes.Axes: A plot of the efficient frontier.
    
    Raises:
    ValueError: If more than two assets are provided.
    """
    # Check if only two assets are provided.
    if er.shape[0] != 2:
        raise ValueError("plot_ef2 can only plot 2-asset frontiers")

    # Generate portfolio weights combinations.
    weights = [np.array([w, 1 - w]) for w in np.linspace(0, 1, n_points)]

    # Calculate returns and volatilities for each weight combination.
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]

    # Create a DataFrame and plot the efficient frontier.
    ef = pd.DataFrame({"Returns": rets, "Volatility": vols})
    return ef.plot.line(x="Volatility", y="Returns", style=style)

  
 
def minimize_vol(target_return, er, cov):
    """
    Finds the optimal portfolio weights that achieve a target return with the lowest possible volatility.

    Parameters:
    target_return (float): The desired target return.
    er (pd.Series): Expected returns for each asset.
    cov (pd.DataFrame): Covariance matrix for the assets.

    Returns:
    np.ndarray: Optimal asset weights for the target return.
    """
    # Number of assets.
    n = er.shape[0]

    # Initial allocation across portfolios (equal weight).
    init_guess = np.repeat(1/n, n)

    # Bounds for each weight (0 <= weight <= 1).
    bounds = ((0.0, 1.0),) * n

    # Define target return as an equality constraint.
    return_is_target = {
        'type': 'eq',
        'args': (er,),
        'fun': lambda weights, er: target_return - portfolio_return(weights, er)
    }

    # Constraint: weights must sum to 1.
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }

    # Use SLSQP method to minimize volatility.
    results = minimize(portfolio_vol, 
                       init_guess, 
                       args=(cov,),
                       method="SLSQP",
                       options={'disp': False},
                       constraints=(return_is_target, weights_sum_to_1),
                       bounds=bounds)
    
    # Return the optimized weights.
    return results.x
    

    
    

def get_ffme_returns():
    """
    Loads the Fama-French Dataset for returns of the top and bottom deciles by market cap.

    Returns:
    pd.DataFrame: Returns of SmallCap and LargeCap stocks.
    """
    file_path = '/content/invest_ml/data/Portfolios_Formed_on_ME_monthly_EW.csv'
    rets = pd.read_csv(file_path, header=0, index_col=0, na_values=99.99)
    rets = rets[['Lo 10', 'Hi 10']]
    rets.columns = ['SmallCap', 'LargeCap']
    rets = rets / 100
    rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period('M')
    return rets


def get_hfi_returns():
    """
    Loads and formats the EDHEC Hedge Fund Index Returns, which contain various hedge fund strategies.

    Returns:
    pd.DataFrame: Returns of different hedge fund strategies.
    """
    file_path = '/content/invest_ml/data/edhec-hedgefundindices.csv'
    hfi = pd.read_csv(file_path, header=0, index_col=0, parse_dates=True)
    hfi = hfi / 100
    hfi.index = hfi.index.to_period('M')
    return hfi

def get_ind_returns():
    """
    Loads and formats the monthly returns of 30 different industry portfolios from 1926 onward.

    Returns:
    pd.DataFrame: Monthly returns of 30 industry portfolios.
    """
    file_path = '/content/invest_ml/data/ind30_m_vw_rets.csv'
    ind = pd.read_csv(file_path, header=0, index_col=0, parse_dates=True)
    ind = ind / 100
    ind.index = pd.to_datetime(ind.index, format='%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind
  

def get_ind_size():
    """
    
    """
    file_path = '/content/ind30_m_size.csv'
    ind = pd.read_csv(file_path, header=0, index_col=0, parse_dates=True)
    ind.index = pd.to_datetime(ind.index, format='%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def get_ind_nfirms():
    """
   
    """
    file_path = '/content/ind30_m_nfirms.csv'
    ind = pd.read_csv(file_path, header=0, index_col=0, parse_dates=True)
    ind.index = pd.to_datetime(ind.index, format='%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def optimal_weights(n_points, er, cov):
    """
    Generates a list of weights to run the optimizer to minimize volatility across a range of target returns.

    Parameters:
    n_points (int): Number of points for which to calculate weights.
    er (pd.Series): Expected returns for each asset.
    cov (pd.DataFrame): Covariance matrix for the assets.

    Returns:
    list: List of optimal asset weights for each target return.
    """
    # Generate a range of target returns.
    target_rs = np.linspace(er.min(), er.max(), n_points)

    # Find the optimal weights for each target return.
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights


def msr(riskfree_rate, er, cov):
    """
    Finds the weights of the portfolio that give the maximum Sharpe ratio.

    Parameters:
    riskfree_rate (float): The risk-free rate.
    er (pd.Series): Expected returns for each asset.
    cov (pd.DataFrame): Covariance matrix for the assets.

    Returns:
    np.ndarray: Optimal asset weights for maximum Sharpe ratio.
    """
    # Number of assets.
    n = er.shape[0]

    # Initial allocation across portfolios (equal weight).
    init_guess = np.repeat(1/n, n)

    # Bounds for each weight (0 <= weight <= 1).
    bounds = ((0.0, 1.0),) * n

    # Constraint: weights must sum to 1.
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }

    # Define the negative Sharpe ratio as the function to minimize.
    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate) / vol

    # Use SLSQP method to maximize Sharpe ratio.
    results = minimize(neg_sharpe_ratio, 
                       init_guess, 
                       args=(riskfree_rate, er, cov),
                       method="SLSQP",
                       options={'disp': False},
                       constraints=(weights_sum_to_1),
                       bounds=bounds)
    
    # Return the optimized weights.
    return results.x


def gmv(cov):
    """
    Returns the weights of the Global Minimum Volatility (GMV) portfolio based on the covariance matrix.

    The GMV portfolio is the portfolio with the lowest possible volatility based on historical returns. 
    It's a cornerstone concept in modern portfolio theory, focusing solely on minimizing risk, 
    regardless of expected returns.

    Parameters:
    - cov (pd.DataFrame): Covariance matrix of the asset returns.

    Returns:
    - numpy.ndarray: Weights of the assets in the GMV portfolio.
    """
    n = cov.shape[0]  # Number of assets in the covariance matrix.
    # The GMV portfolio is found by using the mean-variance optimization to minimize volatility.
    # Here, we use the mean return of 0 for all assets, implying we're not considering returns in this optimization.
    # This function call effectively finds the portfolio on the efficient frontier with the lowest volatility.
    return msr(0, np.repeat(1, n), cov)


def plot_ef(n_points, er, cov, show_cml=True, style='.-', riskfree_rate=0, show_ew=False, show_gmv=False):
    """
    Plots the efficient frontier for a multi-asset portfolio and optionally the Capital Market Line (CML).

    Parameters:
    n_points (int): Number of points to plot on the frontier.
    er (pd.Series): Expected returns for the assets.
    cov (pd.DataFrame): Covariance matrix for the assets.
    show_cml (bool): Whether to show the Capital Market Line.
    style (str): Style of the plot line.
    riskfree_rate (float): The risk-free rate for the CML.

    Returns:
    matplotlib.axes.Axes: A plot of the efficient frontier.
    """
    # Generate optimal weights across a range of target returns.
    weights = optimal_weights(n_points, er, cov)

    # Calculate returns and volatilities for each weight combination.
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]

    # Create a DataFrame and plot the efficient frontier.
    ef = pd.DataFrame({"Returns": rets, "Volatility": vols})
    ax = ef.plot.line(x="Volatility", y="Returns", style=style)
    
    if show_ew:
        # Calculate the Equal Weight portfolio.
        # It assigns an equal weight to all assets, irrespective of their individual risks or returns.
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        # Display the Equal Weight portfolio on the plot as a single point.
        ax.plot([vol_ew], [r_ew], color='sandybrown', marker='o', markersize=12)

    if show_gmv:
        # Calculate the Global Minimum Volatility portfolio weights.
        # This portfolio aims to achieve the lowest possible risk (volatility).
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv,er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        # Display the GMV portfolio on the plot as a single point.
        ax.plot([vol_gmv], [r_gmv], color='midnightblue', marker='o', markersize=12)

# Optionally plot the Capital Market Line.
    if show_cml:
        ax.set_xlim(left=0)
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)

        # Plot the CML.
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color="chartreuse", marker="o", linestyle="dashed", markersize=12, linewidth=2)
        
    return ax        
