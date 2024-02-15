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
- 'get_total_market_return': Calculates the total market return by aggregating the weighted returns of individual industries. It factors in the capitalization weights based on the size and number of firms in each industry, providing a comprehensive view of the market's performance.

- 'get_ind_nfirms': Retrieves the number of firms across different industries. This function is essential for market structure analysis and enables the computation of industry and market capitalization weights necessary for various financial analyses.

- 'get_ind_size': Obtains the total market capitalization size for each industry. This function is crucial for determining the relative size of industries within the market and is used in the calculation of market-cap weighted returns.

- 'run_cppi': Implements the Constant Proportion Portfolio Insurance (CPPI) strategy, dynamically allocating between a risky asset and a safe asset to enforce a predetermined floor value for the portfolio. It can incorporate a dynamic floor adjustment based on a drawdown constraint, providing a robust mechanism for capital preservation in volatile markets.

- 'summary_stats': Aggregates key statistics such as annualized return, volatility, Sharpe ratio, and maximum drawdown, offering a succinct overview of the risk and return profile of an asset or portfolio. This function is instrumental for performance evaluation and comparative analysis of investment strategies.

- 'gbm': Simulates the evolution of stock prices using a Geometric Brownian Motion (GBM) model over a specified number of years and scenarios. The GBM is a stochastic process that models the logarithmic returns of a stock price as normally distributed.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


def compound(r):
    """
    returns the result of compounding the set of returns in r
    """
    return np.expm1(np.log1p(r).sum())
    
    
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
    

    
def get_fff_returns():
    """
    Load the Fama-French Research Factor Monthly Dataset
    """
    rets = pd.read_csv("data/F-F_Research_Data_Factors_m.csv",
                       header=0, index_col=0, na_values=-99.99)/100
    rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period('M')
    return rets
    

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
    Returns sizes of the industry (market cap)
    """
    file_path = '/content/invest_ml/data/ind30_m_size.csv'
    ind = pd.read_csv(file_path, header=0, index_col=0, parse_dates=True)
    ind.index = pd.to_datetime(ind.index, format='%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def get_ind_nfirms():
    """
    Returns number of firms in the industry
    """
    file_path = '/content/invest_ml/data/ind30_m_nfirms.csv'
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

def get_total_market_return():
    '''
    Calculate the total market return based on individual industry returns, the number of firms, and market capitalization sizes.
    
    Returns:
    - A pandas Series representing the weighted total market return over time.
    '''

    # Retrieve the number of firms for each industry
    ind_nfirms = get_ind_nfirms()
    
    # Retrieve the size (market capitalization) for each industry
    ind_size = get_ind_size()
    
    # Calculate the market capitalization for each industry by multiplying the number of firms by their size
    ind_mktcap = ind_nfirms * ind_size
    
    # Calculate the combined market capitalization across all industries for each time period
    joint_mktcap = ind_mktcap.sum(axis='columns')
    
    # Calculate the capitalization weight for each industry by dividing each industry's market cap by the total market cap
    ind_capweight = ind_mktcap.divide(joint_mktcap, axis='rows')
    
    # Calculate the total market return by summing the product of industry returns and their capitalization weights
    # Note: 'ind_return' seems to be a global variable or previously defined data that holds the industry returns
    total_market_return = (ind_capweight * ind_return).sum(axis='columns')
    
    # Return the total market return as a pandas Series
    return total_market_return


def run_cppi(risky_r, safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03, drawdown=None):
    """
    Run a backtest of the CPPI strategy, given a set of returns for the risky asset.
    Returns a dictionary containing: Asset Value History, Risk Budget History, Risky Weight History.
    
    Parameters:
    - risky_r: Returns of the risky asset as a pd.Series or pd.DataFrame.
    - safe_r: Returns of the safe asset as a pd.DataFrame (optional).
    - m: Multiplier applied to the cushion (excess over the floor).
    - start: The starting value of the investment.
    - floor: The minimum acceptable value of the investment, as a percentage of the start value.
    - riskfree_rate: The risk-free rate of return, per period.
    - drawdown: The drawdown adjusts the floor based on the historical peak of the portfolio.
    
    Returns:
    - A dictionary containing the account value, cushion history, and risky weight history.
    """
    # Initialize the CPPI parameters
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start * floor
    peak = start

    # Convert risky_r to a DataFrame if it's a pd.Series
    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns=["R"])
    
    # If safe_r is not provided, create it and set all values to the risk-free rate
    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate/12
            
    # set up some DataFrames for saving intermediate values
    account_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):

        if drawdown is not None:
          peak = np.maximum(peak, account_value)
          floor_value = peak*(1 - drawdown)
          
        # Calculate the cushion as the current account value minus the floor value
        cushion = (account_value - floor_value) / account_value
        risky_w = m * cushion  # Apply the multiplier to the cushion
        risky_w = np.minimum(risky_w, 1)  # The risky weight cannot be more than 100%
        risky_w = np.maximum(risky_w, 0)  # The risky weight cannot be less than 0%
        safe_w = 1 - risky_w  # The safe weight is the remainder of the risky weight

        # Calculate the asset allocation
        risky_alloc = account_value * risky_w
        safe_alloc = account_value * safe_w

        # Update the account value at the end of this step
        account_value = risky_alloc * (1 + risky_r.iloc[step]) + safe_alloc * (1 + safe_r.iloc[step])

        # Save the histories for analysis and plotting
        cushion_history.iloc[step]= cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step]= account_value
    
    # Calculate the Risky Wealth only
    risky_wealth = start * (1+risky_r).cumprod()
    
    # Create the backtest result dictionary
    backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth,
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m": m,
        "start": start,
        "floor": floor,
        "risky_r": risky_r,
        "safe_r": safe_r
    }
    
    return backtest_result


def summary_stats(r,riskfree_rate=0.03):
    '''
    Computes and aggregates critical statistics such as annualized return, volatility, Sharpe ratio, 
    and maximum drawdown. These metrics are vital for assessing the performance and risk profile of 
    our investment strategies.
    Return a DataFrame that contains aggregated summary stats for the returns in the columns or r
    '''
    ann_r = r.aggregate(annualize_rets, periods_per_year=12)
    ann_vol = r.aggregate(annualize_vol, periods_per_year=12)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=12)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min() )
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(cvar_historic)
    return pd.DataFrame({
        "Annualized Return": ann_r, 
        "Annualized Vol": ann_vol,
        "Skewness": skew, 
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })
    
    
def gbm(n_years = 10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0, prices=True):
    """
    Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices through Monte Carlo
    :param n_years:  The number of years to generate data for
    :param n_paths: The number of scenarios/trajectories
    :param mu: Annualized Drift, e.g. Market Return
    :param sigma: Annualized Volatility
    :param steps_per_year: granularity of the simulation
    :param s_0: initial value
    :return: a numpy array of n_paths columns and n_years*steps_per_year rows
    """
    # Derive per-step Model Parameters from User Specifications
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year) + 1
    # the standard way ...
    # rets_plus_1 = np.random.normal(loc=mu*dt+1, scale=sigma*np.sqrt(dt), size=(n_steps, n_scenarios))
    # without discretization error ...
    rets_plus_1 = np.random.normal(loc=(1+mu)**dt, scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    ret_val = s_0*pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1-1
    return ret_val


import matplotlib.pyplot as plt
from IPython.display import display, clear_output

def show_gbm(n_scenarios, mu, sigma):
  '''
  Draws the results of a stock price evolution under a Geometric Brownian Motion model
  '''
  clear_output(wait=True)
  s_0 = 100
  prices = gbm(n_scenarios=n_scenarios, mu=mu, sigma=sigma, s_0=s_0)
  
  ax=prices.plot(legend=False, 
                   color='indianred',
                   style=':',
                   alpha=0.5,
                   figsize=(12,5))
  ax.axhline(y=s_0, ls=':', color='black')
  ax.plot(0, s_0, marker='o', color='darkred', alpha=0.2)  
  plt.show()

'''

import subprocess
import sys

def install_packages():
    """Installs packages."""
    packages = ['dash', 'dash_bootstrap_components']  # Add any other packages you need
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
    
'''
'''
def show_gbm_dash(n_scenarios, mu, sigma):
    """
    Creates and runs a Dash application to visualize the simulation of stock price evolution
    under a Geometric Brownian Motion model.

    Parameters:
    - n_scenarios: The number of scenarios to simulate.
    - mu: The annualized expected return of the stock.
    - sigma: The annualized volatility of the stock returns.
    """
    # stylesheet with the .dbc class from dash-bootstrap-templates library
    dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"

    # Create a new Dash application
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY, dbc_css])

    # Define the layout of the application
    app.layout = html.Div([
        html.Label("Select Number of Scenarios", htmlFor="n_scenarios"),
        dcc.Slider(id='n_scenarios', min=0, max=n_scenarios, step=1, value=n_scenarios, marks=None, tooltip={"placement": "bottom", "always_visible": True}),
        html.Label("Select Avg Return", htmlFor="mu"),
        dcc.Slider(id='mu', min=0, max=0.2, step=0.01, value=mu, marks=None, tooltip={"placement": "bottom", "always_visible": True}),
        html.Label("Select Volatility", htmlFor="sigma"),
        dcc.Slider(id='sigma', min=0, max=0.3, step=0.01, value=sigma, marks=None, tooltip={"placement": "bottom", "always_visible": True}),
        dcc.Graph(id='graph')
    ])


    # Define the callback function that updates the graph
    @app.callback(
        Output('graph', 'figure'),
        [Input('n_scenarios', 'value'), Input('mu', 'value'), Input('sigma', 'value')]
    )
    def update_graph(n_scenarios, mu, sigma):
        s_0 = 100
        prices = gbm(n_scenarios=n_scenarios, mu=mu, sigma=sigma, s_0=s_0)
        df = pd.DataFrame(prices)

        # Create a line chart
        fig = px.line(df, y=df.columns, x=df.index, title='Geometric Brownian Motion')

        # Update line properties for all traces
        fig.update_traces(line=dict(color='#cd5c5c', dash='dot', width=0.9))

        # Update x-axis and y-axis titles
        fig.update_xaxes(title_text='Periods')
        fig.update_yaxes(title_text='Price')

        # Add a horizontal line representing the initial stock price
        fig.add_shape(
            type='line',
            x0=0, x1=1, xref='paper',
            y0=s_0, y1=s_0, yref='y',
            line=dict(color='Black', dash='dot')
        )

        # Add a marker for the initial stock price
        fig.add_trace(go.Scatter(
            x=[0],
            y=[s_0],
            mode='markers',
            marker=dict(color='darkred', size=10, opacity=0.2)
        ))

        # Hide the legend
        fig.update_layout(showlegend=False, plot_bgcolor='rgba(211,211,211,0.1)')

        return fig

    # Start the application
    app.run_server(debug=True)
'''

def discount(t, r):
    """
    Compute the price of a pure discount bond that pays a dollar at time period t
    and r is the per-period interest rate
    returns a |t| x |r| Series or DataFrame
    r can be a float, Series or DataFrame
    returns a DataFrame indexed by t
    """
    discounts = pd.DataFrame([(r+1)**-i for i in t])
    discounts.index = t
    return discounts

def pv(flows, r):
    """
    Compute the present value of a sequence of cash flows given by the time (as an index) and amounts
    r can be a scalar, or a Series or DataFrame with the number of rows matching the num of rows in flows
    """
    dates = flows.index
    discounts = discount(dates, r)
    return discounts.multiply(flows, axis='rows').sum()
    


def funding_ratio(assets, liabilities, r):
    """
    Computes the funding ratio of a series of liabilities, based on an interest rate and current value of assets
    """
    return assets/pv(liabilities, r)
    
    
def inst_to_ann(r):
    """
    Convert an instantaneous interest rate to an annual interest rate
    """
    return np.expm1(r)

def ann_to_inst(r):
    """
    Convert an instantaneous interest rate to an annual interest rate
    """
    return np.log1p(r)

def cir(n_years = 10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05, steps_per_year=12, r_0=None):
    """
    Generate random interest rate evolution over time using the CIR model
    b and r_0 are assumed to be the annualized rates, not the short rate
    and the returned values are the annualized rates as well
    """
    if r_0 is None: r_0 = b 
    r_0 = ann_to_inst(r_0)
    dt = 1/steps_per_year
    num_steps = int(n_years*steps_per_year) + 1 # because n_years might be a float
    
    shock = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0

    ## For Price Generation
    h = math.sqrt(a**2 + 2*sigma**2)
    prices = np.empty_like(shock)
    ####

    def price(ttm, r):
        _A = ((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        _B = (2*(math.exp(h*ttm)-1))/(2*h + (h+a)*(math.exp(h*ttm)-1))
        _P = _A*np.exp(-_B*r)
        return _P
    prices[0] = price(n_years, r_0)
    ####
    
    for step in range(1, num_steps):
        r_t = rates[step-1]
        d_r_t = a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step]
        rates[step] = abs(r_t + d_r_t)
        # generate prices at time t as well ...
        prices[step] = price(n_years-step*dt, rates[step])

    rates = pd.DataFrame(data=inst_to_ann(rates), index=range(num_steps))
    ### for prices
    prices = pd.DataFrame(data=prices, index=range(num_steps))
    ###
    return rates, prices

def bond_cash_flows(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12):
    """
    Returns the series of cash flows generated by a bond,
    indexed by the payment/coupon number
    """
    n_coupons = round(maturity*coupons_per_year)
    coupon_amt = principal*coupon_rate/coupons_per_year
    coupons = np.repeat(coupon_amt, n_coupons)
    coupon_times = np.arange(1, n_coupons+1)
    cash_flows = pd.Series(data=coupon_amt, index=coupon_times)
    cash_flows.iloc[-1] += principal
    return cash_flows
    
def bond_price(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12, discount_rate=0.03):
    """
    Computes the price of a bond that pays regular coupons until maturity
    at which time the principal and the final coupon is returned
    This is not designed to be efficient, rather,
    it is to illustrate the underlying principle behind bond pricing!
    If discount_rate is a DataFrame, then this is assumed to be the rate on each coupon date
    and the bond value is computed over time.
    i.e. The index of the discount_rate DataFrame is assumed to be the coupon number
    """
    if isinstance(discount_rate, pd.DataFrame):
        pricing_dates = discount_rate.index
        prices = pd.DataFrame(index=pricing_dates, columns=discount_rate.columns)
        for t in pricing_dates:
            prices.loc[t] = bond_price(maturity-t/coupons_per_year, principal, coupon_rate, coupons_per_year,
                                      discount_rate.loc[t])
        return prices
    else: # base case ... single time period
        if maturity <= 0: return principal+principal*coupon_rate/coupons_per_year
        cash_flows = bond_cash_flows(maturity, principal, coupon_rate, coupons_per_year)
        return pv(cash_flows, discount_rate/coupons_per_year)

def macaulay_duration(flows, discount_rate):
    """
    Computes the Macaulay Duration of a sequence of cash flows, given a per-period discount rate
    """
    discounted_flows = discount(flows.index, discount_rate)*pd.DataFrame(flows)
    weights = discounted_flows/discounted_flows.sum()
    return np.average(flows.index, weights=weights.iloc[:,0])

def match_durations(cf_t, cf_s, cf_l, discount_rate):
    """
    Returns the weight W in cf_s that, along with (1-W) in cf_l will have an effective
    duration that matches cf_t
    """
    d_t = macaulay_duration(cf_t, discount_rate)
    d_s = macaulay_duration(cf_s, discount_rate)
    d_l = macaulay_duration(cf_l, discount_rate)
    return (d_l - d_t)/(d_l - d_s)

def bond_total_return(monthly_prices, principal, coupon_rate, coupons_per_year):
    """
    Computes the total return of a Bond based on monthly bond prices and coupon payments
    Assumes that dividends (coupons) are paid out at the end of the period (e.g. end of 3 months for quarterly div)
    and that dividends are reinvested in the bond
    """
    coupons = pd.DataFrame(data = 0, index=monthly_prices.index, columns=monthly_prices.columns)
    t_max = monthly_prices.index.max()
    pay_date = np.linspace(12/coupons_per_year, t_max, int(coupons_per_year*t_max/12), dtype=int)
    coupons.iloc[pay_date] = principal*coupon_rate/coupons_per_year
    total_returns = (monthly_prices + coupons)/monthly_prices.shift()-1
    return total_returns.dropna()


def bt_mix(r1, r2, allocator, **kwargs):
    """
    Runs a back test (simulation) of allocating between a two sets of returns
    r1 and r2 are T x N DataFrames or returns where T is the time step index and N is the number of scenarios.
    allocator is a function that takes two sets of returns and allocator specific parameters, and produces
    an allocation to the first portfolio (the rest of the money is invested in the GHP) as a T x 1 DataFrame
    Returns a T x N DataFrame of the resulting N portfolio scenarios
    """
    if not r1.shape == r2.shape:
        raise ValueError("r1 and r2 should have the same shape")
    weights = allocator(r1, r2, **kwargs)
    if not weights.shape == r1.shape:
        raise ValueError("Allocator returned weights with a different shape than the returns")
    r_mix = weights*r1 + (1-weights)*r2
    return r_mix


def fixedmix_allocator(r1, r2, w1, **kwargs):
    """
    Produces a time series over T steps of allocations between the PSP and GHP across N scenarios
    PSP and GHP are T x N DataFrames that represent the returns of the PSP and GHP such that:
     each column is a scenario
     each row is the price for a timestep
    Returns an T x N DataFrame of PSP Weights
    """
    return pd.DataFrame(data = w1, index=r1.index, columns=r1.columns)

def terminal_values(rets):
    """
    Computes the terminal values from a set of returns supplied as a T x N DataFrame
    Return a Series of length N indexed by the columns of rets
    """
    return (rets+1).prod()

def terminal_stats(rets, floor = 0.8, cap=np.inf, name="Stats"):
    """
    Produce Summary Statistics on the terminal values per invested dollar
    across a range of N scenarios
    rets is a T x N DataFrame of returns, where T is the time-step (we assume rets is sorted by time)
    Returns a 1 column DataFrame of Summary Stats indexed by the stat name 
    """
    terminal_wealth = (rets+1).prod()
    breach = terminal_wealth < floor
    reach = terminal_wealth >= cap
    p_breach = breach.mean() if breach.sum() > 0 else np.nan
    p_reach = reach.mean() if reach.sum() > 0 else np.nan
    e_short = (floor-terminal_wealth[breach]).mean() if breach.sum() > 0 else np.nan
    e_surplus = (-cap+terminal_wealth[reach]).mean() if reach.sum() > 0 else np.nan
    sum_stats = pd.DataFrame.from_dict({
        "mean": terminal_wealth.mean(),
        "std" : terminal_wealth.std(),
        "p_breach": p_breach,
        "e_short":e_short,
        "p_reach": p_reach,
        "e_surplus": e_surplus
    }, orient="index", columns=[name])
    return sum_stats

def glidepath_allocator(r1, r2, start_glide=1, end_glide=0.0):
    """
    Allocates weights to r1 starting at start_glide and ends at end_glide
    by gradually moving from start_glide to end_glide over time
    """
    n_points = r1.shape[0]
    n_col = r1.shape[1]
    path = pd.Series(data=np.linspace(start_glide, end_glide, num=n_points))
    paths = pd.concat([path]*n_col, axis=1)
    paths.index = r1.index
    paths.columns = r1.columns
    return paths
    

import statsmodels.api as sm
def regress(dependent_variable, explanatory_variables, alpha=True):
    """
    Runs a linear regression to decompose the dependent variable into the explanatory variables
    returns an object of type statsmodel's RegressionResults on which you can call
       .summary() to print a full summary
       .params for the coefficients
       .tvalues and .pvalues for the significance levels
       .rsquared_adj and .rsquared for quality of fit
    """
    if alpha:
        explanatory_variables = explanatory_variables.copy()
        explanatory_variables["Alpha"] = 1
    
    lm = sm.OLS(dependent_variable, explanatory_variables).fit()
    return lm
    

def style_analysis(dependent_variable, explanatory_variables):
    """
    Returns the optimal weights that minimizes the Tracking error between
    a portfolio of the explanatory variables and the dependent variable
    """
    n = explanatory_variables.shape[1]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    solution = minimize(portfolio_tracking_error, init_guess,
                       args=(dependent_variable, explanatory_variables,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    weights = pd.Series(solution.x, index=explanatory_variables.columns)
    return weights

def portfolio_tracking_error(weights, ref_r, bb_r):
    """
    returns the tracking error between the reference returns
    and a portfolio of building block returns held with given weights
    """
    return tracking_error(ref_r, (weights*bb_r).sum(axis=1))