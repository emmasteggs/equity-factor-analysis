#!/usr/bin/env python
# coding: utf-8

# # Equity Factor Analysis: Value vs Growth in the STOXX 600 Index as a proxy for the European Market
# 
# This project constructs value and growth portfolios using price-to-book ratios for the STOXX 600 and examines their performance relative to the market index using month-end data between 2010 and 2025.

# In[31]:


#importing libraries
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns


# ## 1: Import & Clean Data
# 
# Data for month-end adjusted prices, market capitalisations, and price-to-book ratios for STOXX 600 members were downloaded from Bloomberg via BQL and saved as CSVs. The datasets are cleaned, aligned by date and ticker, and converted to the correct formats.

# In[13]:


#load CSVs
prices_df = pd.read_csv('prices_monthly.csv', low_memory = False)
mkt_cap_df = pd.read_csv('market_cap_monthly.csv', low_memory = False)
pb_df = pd.read_csv('price_to_book_monthly.csv', low_memory = False)


# In[14]:


#clean up dataframes

def clean_bloomberg_df(df):
    #remove empty rows & delete index 0
    df = df.dropna(how = 'all')
    df = df.drop(index=0)
    #rename date column & set as index
    df = df.rename(columns={df.columns[0]: 'Date'})
    df = df.set_index('Date')
    #keep remaining NaNs - ticker not in index at that date. Drop 'Equity' from ticker names
    df.columns = df.columns.str.replace(' Equity', '', regex = False)
    #check dataframes are in the right formats and shapes. parse dates to change from strings to datetime index
    df.index = pd.to_datetime(df.index, dayfirst = True)
    #convert number strings to numbers
    df = df.apply(pd.to_numeric, errors = 'coerce')
    return df

prices_df = clean_bloomberg_df(prices_df)
mkt_cap_df = clean_bloomberg_df(mkt_cap_df)
pb_df = clean_bloomberg_df(pb_df)
    
#align dates and tickers
common_dates = prices_df.index.intersection(mkt_cap_df.index).intersection(pb_df.index)
common_tickers = prices_df.columns.intersection(mkt_cap_df.columns).intersection(pb_df.columns)

prices_df = prices_df.loc[common_dates, common_tickers]
mkt_cap_df = mkt_cap_df.loc[common_dates, common_tickers]
pb_df = pb_df.loc[common_dates, common_tickers]


# ## 2. Returns & Index Construction
# 
# Monthly returns are computed from adjusted share price data. Index returns are constructed using index weights derived from market cap data. P/B data is lagged by one month to avoid look ahead bias in portfolio construction.
# 

# In[15]:


#create new dataframes

#create monthly stock returns dataframe
returns_df = prices_df.pct_change()

#lag P/B by one month to avoid look ahead bias
pb_lag_df = pb_df.shift(1)

#create index weights from market caps
weights_df = mkt_cap_df.div(mkt_cap_df.sum(axis=1), axis=0)

#create index returns
returns_index = (weights_df * returns_df).sum(axis=1)


# ## 3. Value and Growth Portfolio Construction
# 
# Stocks are ranked monthly by P/B ratio. The bottom 30% of index constituents are classified as the Value portfolio, and the top 30% make up the Growth portfolio.
# 
# Portfolios are cap-weighted within each bucket and rebalanced monthly.
# REMOVE: The new value and growth dataframes are renormalised and returns are calculated. The value premium can then be calculated.
# 

# In[16]:


#value or growth classification, bottom 30 percentile value, top 30 growth
pb_pct_df = pb_lag_df.rank(axis = 1, pct = True)
value_stocks_df = pb_pct_df <= 0.3
growth_stocks_df = pb_pct_df >= 0.7

#apply cap weights
weights_value_df = weights_df.where(value_stocks_df, 0)
weights_growth_df = weights_df.where(growth_stocks_df, 0)

#renormalise weights to sum to 1
weights_value_df = weights_value_df.div(weights_value_df.sum(axis=1), axis=0)
weights_growth_df = weights_growth_df.div(weights_growth_df.sum(axis=1), axis=0)

#calculate portfolio returns for value and growth
returns_value = (weights_value_df * returns_df).sum(axis=1)
returns_growth = (weights_growth_df * returns_df).sum(axis=1)

#calculate value premium as monthly return spread between value and growth
value_premium = returns_value - returns_growth

#combine returns into a single dataframe
portfolio_returns_df = pd.DataFrame({
    'Value': returns_value,
    'Growth': returns_growth,
    'Index': returns_index
})


# ## 4. Portfolio Turnover
# 
# Monthly portfolio turnover is calculated as the sum of absolute weights changes between rebalances. This is important to consider as high turnover stratgies imply high transaction costs that can erode returns.
# 

# In[22]:


#calculate turnover function from weights dataframes

def calculate_turnover(weights):
    return weights.diff().abs().sum(axis=1)

turnover_value = calculate_turnover(weights_value_df)
turnover_growth = calculate_turnover(weights_growth_df)
turnover_index = calculate_turnover(weights_df)

print(f"Average monthly turnover — Value: {turnover_value.mean():.1%} | Growth: {turnover_growth.mean():.1%} | Index: {turnover_index.mean():.1%}")


# In[ ]:


## 5. Performance Metrics

Standard performance metrics are computed across the full sample period. We assume the Sharpe ratio uses a 0% risk-free rate, whereas we would use the relevant short-term rate (e.g. EURIBOR) in practice.

REMOVE: A number of useful performance metrics are computed using functions, including annualised returns, volatility, Sharpe ratios, and max drawdowns. These are 


# In[39]:


#define performance metric functions from returns

def returns_annualised(r, periods_per_year = 12):
    r = r.dropna()
    return (1 + r).prod() ** (periods_per_year/len(r)) - 1

def volatility_annualised(r, periods_per_year = 12):
    r = r.dropna()
    return r.std() * np.sqrt(periods_per_year)

def sharpe_ratio(r, r_f = 0.0, periods_per_year = 12):
    r = r.dropna()
    ret_annual = returns_annualised(r, periods_per_year)
    vol_annual = volatility_annualised(r, periods_per_year)
    if vol_annual != 0:
        return (ret_annual - r_f) / vol_annual
    else:
        return np.nan

def max_drawdown(r):
    r = r.dropna()
    cumulative = (1 + r).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = cumulative / rolling_max - 1
    return drawdown.min()

def information_ratio(r_portfolio, r_benchmark, periods_per_year = 12):
    active = (r_portfolio - r_benchmark).dropna()
    tracking_error = active.std() * np.sqrt(periods_per_year)
    active_return = active.mean() * periods_per_year
    if tracking_error != 0:
        return active_return / tracking_error
    else:
        return np.nan

#performance metrics in one function

def performance_summary(r, r_benchmark = None, r_f = 0.0, periods_per_year = 12):
    metrics = {
        'Annualised Return': returns_annualised(r, periods_per_year),
        'Annualised Volatility': volatility_annualised(r, periods_per_year),
        'Sharpe Ratio': sharpe_ratio(r, r_f, periods_per_year),
        'Max Drawdown': max_drawdown(r)
    }
    if r_benchmark is not None:
        metrics['Information Ratio'] = information_ratio(r, r_benchmark, periods_per_year)
    return pd.Series(metrics)

#create performance summary using index as benchmark

metrics_value = performance_summary(returns_value, returns_index)
metrics_growth = performance_summary(returns_growth, returns_index)
metrics_index = performance_summary(returns_index)

performance_df = pd.DataFrame({
    'Value': metrics_value,
    'Growth': metrics_growth,
    'Index': metrics_index
}).T

print("\nPerformance Summary")
print(performance_df.round(4).to_string())


# ## 6. Alpha, Beta & Statistical Significance
# 
# Calculate Jensen's alpha (annualised) and market beta from OLS regression of portfolio excess returns on index excess returns. Then implement a t-test on the monthly value premium to assess whether the return spread between value and growth is statistically distinguishable from zero.

# In[35]:


#compute alpha and beta function

def calculate_alpha_beta(r_portfolio, r_market, r_f = 0.0, periods_per_year = 12):
    excess_portfolio = (r_portfolio - r_f).dropna()
    excess_market = (r_market - r_f).reindex(excess_portfolio.index).dropna()
    aligned = pd.concat([excess_portfolio, excess_market], axis=1).dropna()
    X = sm.add_constant(aligned.iloc[:, 1])
    model = sm.OLS(aligned.iloc[:, 0], X).fit()
    alpha_monthly = model.params['const']
    beta = model.params.iloc[1]
    alpha_annualised = (1 + alpha_monthly) ** periods_per_year - 1
    output = {
        'Alpha (annualised)': alpha_annualised,
    'Beta': beta,
    'R-squared': model.rsquared,
    'Alpha p-value': model.pvalues['const']
    }
    return output

#find alpha and beta for portfolios

alpha_beta_value = calculate_alpha_beta(returns_value, returns_index)
alpha_beta_growth = calculate_alpha_beta(returns_value, returns_index)

print("\nAlpha & Beta vs STOXX 600 Index")
ab_df = pd.DataFrame({'Value': alpha_beta_value, 'Growth': alpha_beta_growth})
print(ab_df.round(4).to_string())


# In[36]:


#t-test on value premium

premium_clean = value_premium.dropna()
t_stat, p_value = stats.ttest_1samp(premium_clean, 0)
print(f"\nValue Premium T-Test")
print(f"Mean monthly premium : {premium_clean.mean():4f} ({premium_clean.mean()*12:.2%} annualised)")
print(f"T-statistic : {t_stat:.3f}")
print(f"P-value : {p_value:.3f}")
print(f"Statistically significant at 5%: {'Yes' if p_value < 0.05 else 'No'}")


# ## 7. Portfolio Performance Comparisons / Visualisations
# 
# Cumulative returns of value, growth, and the market index are plotted using Plotly Express. We can see Value has performed best over time period (c.1,177%), whilst Growth (c.1,027%) and the Index (c.1,001%) lagged behind. Growth had been the best performing portfolio until mid-2025 before Value rallied thanks to low P/B stocks, such as banks and defence companies, re-rating and increasing earnings.
# 

# In[33]:


#plotting cumulative returns

cum_returns_df = (1 + portfolio_returns_df).cumprod() - 1
cum_returns_long = cum_returns_df.reset_index().melt(id_vars = 'Date', var_name = 'Portfolio', value_name = 'Cumulative Return')

fig = px.line(cum_returns_long, x='Date', y='Cumulative Return', color='Portfolio', title='Cumulative Returns: Value vs Growth vs STOXX 600 Index')
fig.update_layout(yaxis_tickformat='.0%')
fig.show()


# In[43]:


#plotting performance metrics
metrics_to_plot = ['Annualised Return', 'Annualised Volatility', 'Sharpe Ratio', 'Max Drawdown']

color_map = {'Value': 'blue', 'Growth': 'red', 'Index': 'green'}

for metric in metrics_to_plot:
    fig = px.bar(performance_df, y = metric, x = performance_df.index,
                 text = performance_df[metric].round(4),
                 title = f'{metric} Comparison',
                 labels = {'x': 'Portfolio', metric: metric},
                 color = performance_df.index,
                 color_discrete_map = color_map
                )
    fig.update_traces(textposition = 'inside', textfont_color = 'white')
    fig.update_layout(yaxis = dict(title=metric), yaxis_tickformat='.0%')
    fig.show()


# In[50]:


#underwater drawdown chart

def drawdown_series(r):
    cumulative = (1+r.dropna()).cumprod()
    return cumulative / cumulative.cummax() -1

dd_df = pd.DataFrame({
    'Value': drawdown_series(returns_value),
    'Growth': drawdown_series(returns_growth),
    'Index': drawdown_series(returns_index)
})

dd_long = dd_df.reset_index().melt(id_vars = 'Date', var_name = 'Portfolio', value_name = 'Drawdown')

fig = px.line(dd_long, x = 'Date', y = 'Drawdown', color = 'Portfolio', title = 'Drawdown Underwater Curve')
fig.update_layout(yaxis_tickformat='.0%')
fig.show()


# In[45]:


#Calculating and plotting rolling 12 month Sharpe ratios

window = 12

#Rolling Sharpe function
def rolling_sharpe(r, window, r_f = 0.0, periods_per_year = 12):
    rolling_mean = r.rolling(window).mean() * periods_per_year
    rolling_std = r.rolling(window).std() * np.sqrt(periods_per_year)
    rolling_sharpe = (rolling_mean - r_f) / rolling_std
    return rolling_sharpe

#DataFrame for plotting
rolling_sharpe_df = pd.DataFrame({
    'Value': rolling_sharpe(returns_value, window),
    'Growth': rolling_sharpe(returns_growth, window),
    'Index': rolling_sharpe(returns_index, window)
})

rolling_sharpe_long = rolling_sharpe_df.reset_index().melt(id_vars = 'Date', var_name = 'Portfolio', value_name = 'Rolling Sharpe')

#Plot
fig = px.line(rolling_sharpe_long, x='Date', y='Rolling Sharpe', color = 'Portfolio', title = 'Rolling 12-Month Sharpe Ratio')
fig.show()


# In[51]:


#Rolling volatility

window = 12

#Rolling volatility dataframe
rolling_vol_df = pd.DataFrame({
    'Value': returns_value.rolling(window).std() * np.sqrt(12),
    'Growth': returns_growth.rolling(window).std() * np.sqrt(12),
    'Index': returns_index.rolling(window).std() * np.sqrt(12)
})

rolling_vol_long = rolling_vol_df.reset_index().melt(id_vars='Date', var_name='Portfolio', value_name='Rolling Volatility')

#Rolling volatility plot
fig_volatility = px.line(rolling_vol_long, x='Date', y='Rolling Volatility', color='Portfolio', title='Rolling 12-Month Annualized Volatility')
fig_volatility.update_layout(yaxis_tickformat='.0%')
fig_volatility.show()


# In[52]:


#Rolling annualised returns

window = 12

#rolling returns DataFrame
rolling_returns_df = pd.DataFrame({
    'Value': (1 + returns_value).rolling(window).apply(np.prod, raw=True) - 1,
    'Growth': (1 + returns_growth).rolling(window).apply(np.prod, raw=True) - 1,
    'Index': (1 + returns_index).rolling(window).apply(np.prod, raw=True) - 1
})

rolling_returns_long = (rolling_returns_df.reset_index().melt(id_vars='Date', var_name='Portfolio', value_name='Rolling Return'))

#Plot
fig = px.line(rolling_returns_long, x='Date', y='Rolling Return', color='Portfolio', title='Rolling 12-Month Returns')
fig.update_layout(yaxis_tickformat='.0%')
fig.show()


# In[63]:


#Correlations

#Plot portfolio returns with seaborn
sns.heatmap(portfolio_returns_df.corr(), annot = True, cmap = 'RdBu_r', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Portfolio Returns')
plt.show()


# In[65]:


#Rolling correlations to see how they've changed over time

window = 12

#DataFrame of rolling correlations between the three indexes
rolling_corr_df = pd.DataFrame({
    'Value vs Growth': returns_value.rolling(window).corr(returns_growth),
    'Value vs Index': returns_value.rolling(window).corr(returns_index),
    'Growth vs Index': returns_growth.rolling(window).corr(returns_index)
})

rolling_corr_long = rolling_corr_df.reset_index().melt(id_vars='Date', var_name='Pair', value_name='Rolling Correlation')

#Plot
fig = px.line(rolling_corr_long, x='Date', y='Rolling Correlation', color='Pair',
              title='Rolling 12-Month Correlations Between Portfolios',color_discrete_map=color_map)
fig.show()


# ## 8. Conclusion
# 
# In the STOXX 600 over the 2010-2025 sample period, the value portfolio (bottom 30% P/B stocks) outperformed both the growth portfolio (top 30% P/B stocks) and the cap-weighted index on a cumulative basis.
# 
# The key takeaways from this project are:
# • Value generated positive alpha vs the index on a market-beta-adjusted basis, consistent with the long-run value premium documented in Fama & French. This was calculated as xxx.
# • The value premium was not consistent across the whole cycle. Growth outperformed materially during the low interest rate environment between 2015 and 2021, when long duration assets benefited from falling discount rates.
# • Value saw a sharp recovery from 2022 onwards, driven by rate normalisation and earnings re-ratings in low P/B sectors, i.e. financials, energy, and defence.
# • The t-test on the monthly value premium indicates whether this outperformance is statistically significant or within the range of sampling noise.
# • Turnover figures highlight the practical constraint that monthly rebalancing is costly. In live implementation transaction cost modelling would be required.
# 
# Limitations of the analysis: Only a single valuation metric, P/B, is used and does not control for sector, country, or quality exposures. A more complete factor model would incorporate additional signals and risk controls.
