# Module-11-Project

Forecasting Net Prophet

You’re a growth analyst at MercadoLibre. With over 200 million users, MercadoLibre is the most popular e-commerce site in Latin America. You've been tasked with analyzing the company's financial and user data in clever ways to make the company grow. So, you want to find out if the ability to predict search traffic can translate into the ability to successfully trade the stock.

Instructions

This section divides the instructions for this Challenge into four steps and an optional fifth step, as follows:

Step 1: Find unusual patterns in hourly Google search traffic

Step 2: Mine the search traffic data for seasonality

Step 3: Relate the search traffic to stock price patterns

Step 4: Create a time series model with Prophet

Step 5 (optional): Forecast revenue by using time series models

The following subsections detail these steps.

Step 1: Find Unusual Patterns in Hourly Google Search Traffic

The data science manager asks if the Google search traffic for the company links to any financial events at the company. Or, does the search traffic data just present random noise? To answer this question, pick out any unusual patterns in the Google search data for the company, and connect them to the corporate financial events.

To do so, complete the following steps:

Read the search data into a DataFrame, and then slice the data to just the month of May 2020. (During this month, MercadoLibre released its quarterly financial results.) Use hvPlot to visualize the results. Do any unusual patterns exist?

Calculate the total search traffic for the month, and then compare the value to the monthly median across all months. Did the Google search traffic increase during the month that MercadoLibre released its financial results?

Step 2: Mine the Search Traffic Data for Seasonality

Marketing realizes that they can use the hourly search data, too. If they can track and predict interest in the company and its platform for any time of day, they can focus their marketing efforts around the times that have the most traffic. This will get a greater return on investment (ROI) from their marketing budget.

To that end, you want to mine the search traffic data for predictable seasonal patterns of interest in the company. To do so, complete the following steps:

Group the hourly search data to plot the average traffic by the day of the week (for example, Monday vs. Friday).

Using hvPlot, visualize this traffic as a heatmap, referencing the index.hour as the x-axis and the index.dayofweek as the y-axis. Does any day-of-week effect that you observe concentrate in just a few hours of that day?

Group the search data by the week of the year. Does the search traffic tend to increase during the winter holiday period (weeks 40 through 52)?

Step 3: Relate the Search Traffic to Stock Price Patterns

You mention your work on the search traffic data during a meeting with people in the finance group at the company. They want to know if any relationship between the search data and the company stock price exists, and they ask if you can investigate.

To do so, complete the following steps:

Read in and plot the stock price data. Concatenate the stock price data to the search data in a single DataFrame.

Market events emerged during the year of 2020 that many companies found difficult. But, after the initial shock to global financial markets, new customers and revenue increased for e-commerce platforms. Slice the data to just the first half of 2020 (2020-01 to 2020-06 in the DataFrame), and then use hvPlot to plot the data. Do both time series indicate a common trend that’s consistent with this narrative?

Create a new column in the DataFrame named “Lagged Search Trends” that offsets, or shifts, the search traffic by one hour. Create two additional columns:

“Stock Volatility”, which holds an exponentially weighted four-hour rolling average of the company’s stock volatility

“Hourly Stock Return”, which holds the percent change of the company's stock price on an hourly basis

Review the time series correlation, and then answer the following question: Does a predictable relationship exist between the lagged search traffic and the stock volatility or between the lagged search traffic and the stock price returns?

Step 4: Create a Time Series Model with Prophet

Now, you need to produce a time series model that analyzes and forecasts patterns in the hourly search data. To do so, complete the following steps:

Set up the Google search data for a Prophet forecasting model.

After estimating the model, plot the forecast. How's the near-term forecast for the popularity of MercadoLibre?

Plot the individual time series components of the model to answer the following questions:

What time of day exhibits the greatest popularity?

Which day of the week gets the most search traffic?

What's the lowest point for search traffic in the calendar year?

Step 5 (Optional): Forecast Revenue by Using Time Series Models

A few weeks after your initial analysis, the finance group follows up to find out if you can help them solve a different problem. Your fame as a growth analyst in the company continues to grow!

Specifically, the finance group wants a forecast of the total sales for the next quarter. This will dramatically increase their ability to plan budgets and to help guide expectations for the company investors.

To do so, complete the following steps:

Read in the daily historical sales (that is, revenue) figures, and then apply a Prophet model to the data.

Interpret the model output to identify any seasonal patterns in the company's revenue. For example, what are the peak revenue days? (Mondays? Fridays? Something else?)

Produce a sales forecast for the finance group. Give them a number for the expected total sales in the next quarter. Include the best- and worst-case scenarios to help them make better plans.

Install and import the required libraries and dependencies

# Install the required libraries
from IPython.display import clear_output
try:
    !pip install pystan
    !pip install prophet
    !pip install hvplot
    !pip install holoviews
except:
  print("Error installing libraries")
finally:
  clear_output()
  print("Libraries successfully installed")
Libraries successfully installed

# Import the required libraries and dependencies
import pandas as pd
import holoviews as hv
from prophet import Prophet
import hvplot.pandas
import datetime as dt
%matplotlib inline
​
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
Step 1: Find Unusual Patterns in Hourly Google Search Traffic

The data science manager asks if the Google search traffic for the company links to any financial events at the company. Or, does the search traffic data just present random noise? To answer this question, pick out any unusual patterns in the Google search data for the company, and connect them to the corporate financial events.

To do so, complete the following steps:

Read the search data into a DataFrame, and then slice the data to just the month of May 2020. (During this month, MercadoLibre released its quarterly financial results.) Use hvPlot to visualize the results. Do any unusual patterns exist?

Calculate the total search traffic for the month, and then compare the value to the monthly median across all months. Did the Google search traffic increase during the month that MercadoLibre released its financial results?

Step 1: Read the search data into a DataFrame, and then slice the data to just the month of May 2020. (During this month, MercadoLibre released its quarterly financial results.) Use hvPlot to visualize the results. Do any unusual patterns exist?

# Upload the "google_hourly_search_trends.csv" file into Colab, then store in a Pandas DataFrame
# Set the "Date" column as the Datetime Index.
​
df_mercado_trends = pd.read_csv("Resources/google_hourly_search_trends.csv",
                                index_col="Date",
                                parse_dates=True,
                                infer_datetime_format=True)
​
# Review the first and last five rows of the DataFrame
df_mercado_trends
Search Trends
Date	
2016-06-01 00:00:00	97
2016-06-01 01:00:00	92
2016-06-01 02:00:00	76
2016-06-01 03:00:00	60
2016-06-01 04:00:00	38
...	...
2020-09-07 20:00:00	71
2020-09-07 21:00:00	83
2020-09-07 22:00:00	96
2020-09-07 23:00:00	97
2020-09-08 00:00:00	96
37106 rows × 1 columns


# Review the data types of the DataFrame using the info function
df_mercado_trends.info()
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 37106 entries, 2016-06-01 00:00:00 to 2020-09-08 00:00:00
Data columns (total 1 columns):
 #   Column         Non-Null Count  Dtype
---  ------         --------------  -----
 0   Search Trends  37106 non-null  int64
dtypes: int64(1)
memory usage: 579.8 KB

# Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')
​
# Slice the DataFrame to just the month of May 2020
df_may_2020 = df_mercado_trends.loc["2020-05"]
​
# Use hvPlot to visualize the data for May 2020
df_may_2020.hvplot()
ImageImage
Step 2: Calculate the total search traffic for the month, and then compare the value to the monthly median across all months. Did the Google search traffic increase during the month that MercadoLibre released its financial results?

# Calculate the sum of the total search traffic for May 2020
traffic_may_2020 = df_may_2020.sum()
​
# View the traffic_may_2020 value
traffic_may_2020
Search Trends    38181
dtype: int64

# Calcluate the monhtly median search traffic across all months 
# Group the DataFrame by index year and then index month, chain the sum and then the median functions
median_monthly_traffic = df_mercado_trends.groupby([df_mercado_trends.index.year,df_mercado_trends.index.month]).sum().median()
​
# View the median_monthly_traffic value
median_monthly_traffic
Search Trends    35172.5
dtype: float64

# Compare the seach traffic for the month of May 2020 to the overall monthly median value
print(f"The search traffic for the month of May 2020 was {traffic_may_2020[0]}.")
print(f"The overall monthly median value of search traffic was {median_monthly_traffic[0]}.")
The search traffic for the month of May 2020 was 38181.
The overall monthly median value of search traffic was 35172.5.
Answer the following question:
Question: Did the Google search traffic increase during the month that MercadoLibre released its financial results?

Answer: Yes the search traffic for May increased compared to median values over other months.

Step 2: Mine the Search Traffic Data for Seasonality

Marketing realizes that they can use the hourly search data, too. If they can track and predict interest in the company and its platform for any time of day, they can focus their marketing efforts around the times that have the most traffic. This will get a greater return on investment (ROI) from their marketing budget.

To that end, you want to mine the search traffic data for predictable seasonal patterns of interest in the company. To do so, complete the following steps:

Group the hourly search data to plot the average traffic by the day of the week (for example, Monday vs. Friday).

Using hvPlot, visualize this traffic as a heatmap, referencing the index.hour as the x-axis and the index.dayofweek as the y-axis. Does any day-of-week effect that you observe concentrate in just a few hours of that day?

Group the search data by the week of the year. Does the search traffic tend to increase during the winter holiday period (weeks 40 through 52)?

Step 1: Group the hourly search data to plot the average traffic by the day of the week (for example, Monday vs. Friday).

# Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')
​
# Group the hourly search data to plot (use hvPlot) the average traffic by the day of week 
df_mercado_trends.groupby([df_mercado_trends.index.dayofweek]).mean().hvplot(
    title = "Average Traffic by Day of the Week",
    xlabel = "Day of the Week")
ImageImage
Step 2: Using hvPlot, visualize this traffic as a heatmap, referencing the index.hour as the x-axis and the index.dayofweek as the y-axis. Does any day-of-week effect that you observe concentrate in just a few hours of that day?

# Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')
​
# Use hvPlot to visualize the hour of the day and day of week search traffic as a heatmap.
df_mercado_trends.hvplot.heatmap(
    title = "Search Traffic by Hour and Day",
    x = "index.hour",
    xlabel = "Hour of the Day",
    y = "index.dayofweek",
    ylabel = "Day of the Week",
    C = "Search Trends")
ImageImage
WARNING:param.HeatMapPlot01104: HeatMap element index is not unique,  ensure you aggregate the data before displaying it, e.g. using heatmap.aggregate(function=np.mean). Duplicate index values have been dropped.
Answer the following question:
Question: Does any day-of-week effect that you observe concentrate in just a few hours of that day?

Answer: Search traffic appears to be clustered late at night on many days.

Step 3: Group the search data by the week of the year. Does the search traffic tend to increase during the winter holiday period (weeks 40 through 52)?

# Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')
​
# Group the hourly search data to plot (use hvPlot) the average traffic by the week of the year
df_mercado_trends.groupby(df_mercado_trends.index.isocalendar().week).mean().hvplot().opts(xrotation=90, width=800)
ImageImage
Answer the following question:
Question: Does the search traffic tend to increase during the winter holiday period (weeks 40 through 52)?

Answer: Traffic appears to increase in the winter holiday period but dives off into the final weeks of the year. There is a tremendous spike observed at the beginning of the year after that I suppose to account for returns from the holidays and sales to move leftover holiday inventory.

Step 3: Relate the Search Traffic to Stock Price Patterns

You mention your work on the search traffic data during a meeting with people in the finance group at the company. They want to know if any relationship between the search data and the company stock price exists, and they ask if you can investigate.

To do so, complete the following steps:

Read in and plot the stock price data. Concatenate the stock price data to the search data in a single DataFrame.

Market events emerged during the year of 2020 that many companies found difficult. But, after the initial shock to global financial markets, new customers and revenue increased for e-commerce platforms. Slice the data to just the first half of 2020 (2020-01 to 2020-06 in the DataFrame), and then use hvPlot to plot the data. Do both time series indicate a common trend that’s consistent with this narrative?

Create a new column in the DataFrame named “Lagged Search Trends” that offsets, or shifts, the search traffic by one hour. Create two additional columns:

“Stock Volatility”, which holds an exponentially weighted four-hour rolling average of the company’s stock volatility

“Hourly Stock Return”, which holds the percent change of the company's stock price on an hourly basis

Review the time series correlation, and then answer the following question: Does a predictable relationship exist between the lagged search traffic and the stock volatility or between the lagged search traffic and the stock price returns?

Step 1: Read in and plot the stock price data. Concatenate the stock price data to the search data in a single DataFrame.

# Upload the "mercado_stock_price.csv" file into Colab, then store in a Pandas DataFrame
# Set the "date" column as the Datetime Index.
​
df_mercado_stock = pd.read_csv("Resources/mercado_stock_price.csv",
                               index_col = "date",
                               parse_dates = True,
                               infer_datetime_format = True)
​
# View the first and last five rows of the DataFrame
df_mercado_stock
close
date	
2015-01-02 09:00:00	127.670
2015-01-02 10:00:00	125.440
2015-01-02 11:00:00	125.570
2015-01-02 12:00:00	125.400
2015-01-02 13:00:00	125.170
...	...
2020-07-31 11:00:00	1105.780
2020-07-31 12:00:00	1087.925
2020-07-31 13:00:00	1095.800
2020-07-31 14:00:00	1110.650
2020-07-31 15:00:00	1122.510
48895 rows × 1 columns


# Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')
​
# Use hvPlot to visualize the closing price of the df_mercado_stock DataFrame
df_mercado_stock["close"].hvplot(title = "Closing Price",
                                xlabel = "Date",
                                ylabel = "Closing Price")
ImageImage

# Concatenate the df_mercado_stock DataFrame with the df_mercado_trends DataFrame
# Concatenate the DataFrame by columns (axis=1), and drop and rows with only one column of data
mercado_stock_trends_df = mercado_stock_trends_df = pd.concat([df_mercado_stock, df_mercado_trends], axis=1).dropna()
​
# View the first and last five rows of the DataFrame
mercado_stock_trends_df
close	Search Trends
2016-06-01 09:00:00	135.160	6.0
2016-06-01 10:00:00	136.630	12.0
2016-06-01 11:00:00	136.560	22.0
2016-06-01 12:00:00	136.420	33.0
2016-06-01 13:00:00	136.100	40.0
...	...	...
2020-07-31 11:00:00	1105.780	20.0
2020-07-31 12:00:00	1087.925	32.0
2020-07-31 13:00:00	1095.800	41.0
2020-07-31 14:00:00	1110.650	47.0
2020-07-31 15:00:00	1122.510	53.0
7067 rows × 2 columns

Step 2: Market events emerged during the year of 2020 that many companies found difficult. But, after the initial shock to global financial markets, new customers and revenue increased for e-commerce platforms. Slice the data to just the first half of 2020 (2020-01 to 2020-06 in the DataFrame), and then use hvPlot to plot the data. Do both time series indicate a common trend that’s consistent with this narrative?

# For the combined dataframe, slice to just the first half of 2020 (2020-01 through 2020-06) 
first_half_2020 = mercado_stock_trends_df.loc["2020-01" : "2020-06"]
​
# View the first and last five rows of first_half_2020 DataFrame
first_half_2020
close	Search Trends
2020-01-02 09:00:00	601.085	9.0
2020-01-02 10:00:00	601.290	14.0
2020-01-02 11:00:00	615.410	25.0
2020-01-02 12:00:00	611.400	37.0
2020-01-02 13:00:00	611.830	50.0
...	...	...
2020-06-30 11:00:00	976.170	17.0
2020-06-30 12:00:00	977.500	27.0
2020-06-30 13:00:00	973.230	37.0
2020-06-30 14:00:00	976.500	45.0
2020-06-30 15:00:00	984.930	51.0
807 rows × 2 columns


# Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')
​
# Use hvPlot to visualize the close and Search Trends data
# Plot each column on a separate axes using the following syntax
# `hvplot(shared_axes=False, subplots=True).cols(1)`
first_half_2020.hvplot(shared_axes=False, subplots=True).cols(1)
ImageImage
Answer the following question:

**Question:** Do both time series indicate a common trend that’s consistent with this narrative?
​
**Answer:** There is a strong correlation between search traffic numbers and stock price observable particularly around 3/2020 and right after 5/2020.
Step 3: Create a new column in the DataFrame named “Lagged Search Trends” that offsets, or shifts, the search traffic by one hour. Create two additional columns:

“Stock Volatility”, which holds an exponentially weighted four-hour rolling average of the company’s stock volatility

“Hourly Stock Return”, which holds the percent change of the company's stock price on an hourly basis


# Create a new column in the mercado_stock_trends_df DataFrame called Lagged Search Trends
# This column should shift the Search Trends information by one hour
mercado_stock_trends_df['Lagged Search Trends'] = mercado_stock_trends_df["Search Trends"].shift()

# Create a new column in the mercado_stock_trends_df DataFrame called Stock Volatility
# This column should calculate the standard deviation of the closing stock price return data over a 4 period rolling window
mercado_stock_trends_df['Stock Volatility'] = mercado_stock_trends_df["Stock Volatility"] = mercado_stock_trends_df["close"].pct_change().rolling(window=4).std()

# Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')
​
# Use hvPlot to visualize the stock volatility
mercado_stock_trends_df["Stock Volatility"].hvplot()
ImageImage
Solution Note: Note how volatility spiked, and tended to stay high, during the first half of 2020. This is a common characteristic of volatility in stock returns worldwide: high volatility days tend to be followed by yet more high volatility days. When it rains, it pours.


# Create a new column in the mercado_stock_trends_df DataFrame called Hourly Stock Return
# This column should calculate hourly return percentage of the closing price
mercado_stock_trends_df['Hourly Stock Return'] = mercado_stock_trends_df["close"].pct_change()

# View the first and last five rows of the mercado_stock_trends_df DataFrame
mercado_stock_trends_df
close	Search Trends	Lagged Search Trends	Stock Volatility	Hourly Stock Return
2016-06-01 09:00:00	135.160	6.0	NaN	NaN	NaN
2016-06-01 10:00:00	136.630	12.0	6.0	NaN	0.010876
2016-06-01 11:00:00	136.560	22.0	12.0	NaN	-0.000512
2016-06-01 12:00:00	136.420	33.0	22.0	NaN	-0.001025
2016-06-01 13:00:00	136.100	40.0	33.0	0.006134	-0.002346
...	...	...	...	...	...
2020-07-31 11:00:00	1105.780	20.0	11.0	0.012837	0.006380
2020-07-31 12:00:00	1087.925	32.0	20.0	0.013549	-0.016147
2020-07-31 13:00:00	1095.800	41.0	32.0	0.013295	0.007239
2020-07-31 14:00:00	1110.650	47.0	41.0	0.013001	0.013552
2020-07-31 15:00:00	1122.510	53.0	47.0	0.013566	0.010678
7067 rows × 5 columns

Step 4: Review the time series correlation, and then answer the following question: Does a predictable relationship exist between the lagged search traffic and the stock volatility or between the lagged search traffic and the stock price returns?

# Construct correlation table of Stock Volatility, Lagged Search Trends, and Hourly Stock Return
mercado_stock_trends_df[["Stock Volatility", "Lagged Search Trends", "Hourly Stock Return"]].corr()
Stock Volatility	Lagged Search Trends	Hourly Stock Return
Stock Volatility	1.000000	-0.148938	0.061424
Lagged Search Trends	-0.148938	1.000000	0.017929
Hourly Stock Return	0.061424	0.017929	1.000000
Answer the following question:
Question: Does a predictable relationship exist between the lagged search traffic and the stock volatility or between the lagged search traffic and the stock price returns?

Answer: As expected, yes

Step 4: Create a Time Series Model with Prophet

Now, you need to produce a time series model that analyzes and forecasts patterns in the hourly search data. To do so, complete the following steps:

Set up the Google search data for a Prophet forecasting model.

After estimating the model, plot the forecast. How's the near-term forecast for the popularity of MercadoLibre?

Plot the individual time series components of the model to answer the following questions:

What time of day exhibits the greatest popularity?

Which day of the week gets the most search traffic?

What's the lowest point for search traffic in the calendar year?

Step 1: Set up the Google search data for a Prophet forecasting model.

# Using the df_mercado_trends DataFrame, reset the index so the date information is no longer the index
mercado_prophet_df = df_mercado_trends.reset_index()
​
# Label the columns ds and y so that the syntax is recognized by Prophet
mercado_prophet_df.columns = ["ds", "y"]
​
# Drop an NaN values from the prophet_df DataFrame
mercado_prophet_df = mercado_prophet_df.dropna()
​
# View the first and last five rows of the mercado_prophet_df DataFrame
mercado_prophet_df
ds	y
0	2016-06-01 00:00:00	97
1	2016-06-01 01:00:00	92
2	2016-06-01 02:00:00	76
3	2016-06-01 03:00:00	60
4	2016-06-01 04:00:00	38
...	...	...
37101	2020-09-07 20:00:00	71
37102	2020-09-07 21:00:00	83
37103	2020-09-07 22:00:00	96
37104	2020-09-07 23:00:00	97
37105	2020-09-08 00:00:00	96
37106 rows × 2 columns


# Call the Prophet function, store as an object
model_mercado_trends = Prophet()

# Fit the time-series model.
model_mercado_trends.fit(mercado_prophet_df)
12:27:57 - cmdstanpy - INFO - Chain [1] start processing
12:28:05 - cmdstanpy - INFO - Chain [1] done processing
<prophet.forecaster.Prophet at 0x14783e510>

# Create a future dataframe to hold predictions
# Make the prediction go out as far as 2000 hours (approx 80 days)
future_mercado_trends = model_mercado_trends.make_future_dataframe(periods = 2000, freq = "H")
​
# View the last five rows of the future_mercado_trends DataFrame
future_mercado_trends.tail()
ds
39101	2020-11-30 04:00:00
39102	2020-11-30 05:00:00
39103	2020-11-30 06:00:00
39104	2020-11-30 07:00:00
39105	2020-11-30 08:00:00

# Make the predictions for the trend data using the future_mercado_trends DataFrame
forecast_mercado_trends = model_mercado_trends.predict(future_mercado_trends)
​
# Display the first five rows of the forecast_mercado_trends DataFrame
forecast_mercado_trends.head()
ds	trend	yhat_lower	yhat_upper	trend_lower	trend_upper	additive_terms	additive_terms_lower	additive_terms_upper	daily	...	weekly	weekly_lower	weekly_upper	yearly	yearly_lower	yearly_upper	multiplicative_terms	multiplicative_terms_lower	multiplicative_terms_upper	yhat
0	2016-06-01 00:00:00	44.129250	81.677206	97.529122	44.129250	44.129250	45.429305	45.429305	45.429305	41.452726	...	1.860133	1.860133	1.860133	2.116445	2.116445	2.116445	0.0	0.0	0.0	89.558555
1	2016-06-01 01:00:00	44.130313	77.631485	94.913088	44.130313	44.130313	41.875073	41.875073	41.875073	37.943506	...	1.810049	1.810049	1.810049	2.121518	2.121518	2.121518	0.0	0.0	0.0	86.005386
2	2016-06-01 02:00:00	44.131375	67.095992	83.774069	44.131375	44.131375	31.551566	31.551566	31.551566	27.656533	...	1.768474	1.768474	1.768474	2.126559	2.126559	2.126559	0.0	0.0	0.0	75.682942
3	2016-06-01 03:00:00	44.132438	51.674745	68.449680	44.132438	44.132438	16.284352	16.284352	16.284352	12.417280	...	1.735502	1.735502	1.735502	2.131569	2.131569	2.131569	0.0	0.0	0.0	60.416789
4	2016-06-01 04:00:00	44.133500	35.957867	52.086739	44.133500	44.133500	-0.830504	-0.830504	-0.830504	-4.678139	...	1.711088	1.711088	1.711088	2.136547	2.136547	2.136547	0.0	0.0	0.0	43.302997
5 rows × 22 columns

Step 2: After estimating the model, plot the forecast. How's the near-term forecast for the popularity of MercadoLibre?

# Plot the Prophet predictions for the Mercado trends data
model_mercado_trends.plot(forecast_mercado_trends, xlabel = "Date", ylabel = "Search Trends");

Answer the following question:
Question: How's the near-term forecast for the popularity of MercadoLibre?

Answer: Slightly negative near term with an expected uptick in search trends over time.

Step 3: Plot the individual time series components of the model to answer the following questions:

What time of day exhibits the greatest popularity?

Which day of the week gets the most search traffic?

What's the lowest point for search traffic in the calendar year?


# Set the index in the forecast_mercado_trends DataFrame to the ds datetime column
forecast_mercado_trends = forecast_mercado_trends.set_index(["ds"])
​
# View the only the yhat,yhat_lower and yhat_upper columns from the DataFrame
forecast_mercado_trends[["yhat", "yhat_lower", "yhat_upper"]]
yhat	yhat_lower	yhat_upper
ds			
2016-06-01 00:00:00	89.558555	81.677206	97.529122
2016-06-01 01:00:00	86.005386	77.631485	94.913088
2016-06-01 02:00:00	75.682942	67.095992	83.774069
2016-06-01 03:00:00	60.416789	51.674745	68.449680
2016-06-01 04:00:00	43.302997	35.957867	52.086739
...	...	...	...
2020-11-30 04:00:00	39.283272	31.636279	47.391643
2020-11-30 05:00:00	23.812065	15.218270	32.413586
2020-11-30 06:00:00	11.846318	3.109953	20.197462
2020-11-30 07:00:00	4.574080	-4.788706	13.160264
2020-11-30 08:00:00	2.379109	-6.457345	10.660740
39106 rows × 3 columns

Solutions Note: yhat represents the most likely (average) forecast, whereas yhat_lower and yhat_upper represents the worst and best case prediction (based on what are known as 95% confidence intervals).


# Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')
​
# From the forecast_mercado_trends DataFrame, use hvPlot to visualize
#  the yhat, yhat_lower, and yhat_upper columns over the last 2000 hours 
forecast_mercado_trends[["yhat", "yhat_lower", "yhat_upper"]].iloc[-2000:, :].hvplot(xlabel = "Date")
ImageImage

# Reset the index in the forecast_mercado_trends DataFrame
forecast_mercado_trends = forecast_mercado_trends.reset_index()
​
# Use the plot_components function to visualize the forecast results 
# for the forecast_canada DataFrame 
figures_mercado_trends = model_mercado_trends.plot_components(forecast_mercado_trends)

Answer the following questions:
Question: What time of day exhibits the greatest popularity?

Answer: Midnight

Question: Which day of week gets the most search traffic?

Answer: Tuesday

Question: What's the lowest point for search traffic in the calendar year?

Answer: Mid October

Step 5 (Optional): Forecast Revenue by Using Time Series Models

A few weeks after your initial analysis, the finance group follows up to find out if you can help them solve a different problem. Your fame as a growth analyst in the company continues to grow!

Specifically, the finance group wants a forecast of the total sales for the next quarter. This will dramatically increase their ability to plan budgets and to help guide expectations for the company investors.

To do so, complete the following steps:

Read in the daily historical sales (that is, revenue) figures, and then apply a Prophet model to the data. The daily sales figures are quoted in millions of USD dollars.

Interpret the model output to identify any seasonal patterns in the company's revenue. For example, what are the peak revenue days? (Mondays? Fridays? Something else?)

Produce a sales forecast for the finance group. Give them a number for the expected total sales in the next quarter. Include the best- and worst-case scenarios to help them make better plans.

Step 1: Read in the daily historical sales (that is, revenue) figures, and then apply a Prophet model to the data.

# Upload the "mercado_daily_revenue.csv" file into Colab, then store in a Pandas DataFrame
# Set the "date" column as the DatetimeIndex
# Sales are quoted in millions of US dollars
​
df_mercado_sales = pd.read_csv("Resources/mercado_daily_revenue.csv",
                               index_col="date",
                               parse_dates=True,
                               infer_datetime_format=True)
​
# Review the DataFrame
df_mercado_sales
Daily Sales
date	
2019-01-01	0.626452
2019-01-02	1.301069
2019-01-03	1.751689
2019-01-04	3.256294
2019-01-05	3.732920
...	...
2020-05-10	17.467814
2020-05-11	17.537152
2020-05-12	18.031773
2020-05-13	19.165315
2020-05-14	20.246570
500 rows × 1 columns


# Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')
​
# Use hvPlot to visualize the daily sales figures 
df_mercado_sales.hvplot(xlabel = "Date")
ImageImage

# Apply a Facebook Prophet model to the data.
​
# Set up the dataframe in the neccessary format:
# Reset the index so that date becomes a column in the DataFrame
mercado_sales_prophet_df = df_mercado_sales.reset_index()
​
# Adjust the columns names to the Prophet syntax
mercado_sales_prophet_df.columns = ["ds", "y"]
​
# Visualize the DataFrame
mercado_sales_prophet_df
ds	y
0	2019-01-01	0.626452
1	2019-01-02	1.301069
2	2019-01-03	1.751689
3	2019-01-04	3.256294
4	2019-01-05	3.732920
...	...	...
495	2020-05-10	17.467814
496	2020-05-11	17.537152
497	2020-05-12	18.031773
498	2020-05-13	19.165315
499	2020-05-14	20.246570
500 rows × 2 columns


# Create the model
mercado_sales_prophet_model = Prophet()
​
# Fit the model
mercado_sales_prophet_model.fit(mercado_sales_prophet_df)
13:14:10 - cmdstanpy - INFO - Chain [1] start processing
13:14:10 - cmdstanpy - INFO - Chain [1] done processing
<prophet.forecaster.Prophet at 0x16801fa50>

# Predict sales for 90 days (1 quarter) out into the future.
​
# Start by making a future dataframe
mercado_sales_prophet_future = mercado_sales_prophet_model.make_future_dataframe(periods = 90, freq = "D")
​
# Display the last five rows of the future DataFrame
mercado_sales_prophet_future.tail()
ds
585	2020-08-08
586	2020-08-09
587	2020-08-10
588	2020-08-11
589	2020-08-12

# Make predictions for the sales each day over the next quarter
mercado_sales_prophet_forecast = mercado_sales_prophet_model.predict(mercado_sales_prophet_future)
​
# Display the first 5 rows of the resulting DataFrame
mercado_sales_prophet_forecast.head()
ds	trend	yhat_lower	yhat_upper	trend_lower	trend_upper	additive_terms	additive_terms_lower	additive_terms_upper	weekly	weekly_lower	weekly_upper	multiplicative_terms	multiplicative_terms_lower	multiplicative_terms_upper	yhat
0	2019-01-01	0.132556	-1.645135	2.207300	0.132556	0.132556	0.063973	0.063973	0.063973	0.063973	0.063973	0.063973	0.0	0.0	0.0	0.196528
1	2019-01-02	0.171741	-1.605731	2.048157	0.171741	0.171741	0.083128	0.083128	0.083128	0.083128	0.083128	0.083128	0.0	0.0	0.0	0.254869
2	2019-01-03	0.210927	-1.785374	2.015180	0.210927	0.210927	0.019673	0.019673	0.019673	0.019673	0.019673	0.019673	0.0	0.0	0.0	0.230600
3	2019-01-04	0.250112	-1.853445	2.139120	0.250112	0.250112	-0.058265	-0.058265	-0.058265	-0.058265	-0.058265	-0.058265	0.0	0.0	0.0	0.191847
4	2019-01-05	0.289298	-1.691300	1.977181	0.289298	0.289298	-0.124464	-0.124464	-0.124464	-0.124464	-0.124464	-0.124464	0.0	0.0	0.0	0.164834
Step 2: Interpret the model output to identify any seasonal patterns in the company's revenue. For example, what are the peak revenue days? (Mondays? Fridays? Something else?)

# Use the plot_components function to analyze seasonal patterns in the company's revenue
mercado_sales_prophet_model.plot_components(mercado_sales_prophet_forecast);

Answer the following question:
Question: For example, what are the peak revenue days? (Mondays? Fridays? Something else?)

Answer: Wednesday

Step 3: Produce a sales forecast for the finance group. Give them a number for the expected total sales in the next quarter. Include the best- and worst-case scenarios to help them make better plans.

# Plot the predictions for the Mercado sales
mercado_sales_prophet_model.plot(mercado_sales_prophet_forecast,
                                 include_legend=True,
                                 ylabel = "Total Sales",
                                xlabel = "Date");


# For the mercado_sales_prophet_forecast DataFrame, set the ds column as the DataFrame Index
mercado_sales_prophet_forecast = mercado_sales_prophet_forecast.set_index(["ds"])
​
# Display the first and last five rows of the DataFrame
mercado_sales_prophet_forecast
trend	yhat_lower	yhat_upper	trend_lower	trend_upper	additive_terms	additive_terms_lower	additive_terms_upper	weekly	weekly_lower	weekly_upper	multiplicative_terms	multiplicative_terms_lower	multiplicative_terms_upper	yhat
ds															
2019-01-01	0.132556	-1.645135	2.207300	0.132556	0.132556	0.063973	0.063973	0.063973	0.063973	0.063973	0.063973	0.0	0.0	0.0	0.196528
2019-01-02	0.171741	-1.605731	2.048157	0.171741	0.171741	0.083128	0.083128	0.083128	0.083128	0.083128	0.083128	0.0	0.0	0.0	0.254869
2019-01-03	0.210927	-1.785374	2.015180	0.210927	0.210927	0.019673	0.019673	0.019673	0.019673	0.019673	0.019673	0.0	0.0	0.0	0.230600
2019-01-04	0.250112	-1.853445	2.139120	0.250112	0.250112	-0.058265	-0.058265	-0.058265	-0.058265	-0.058265	-0.058265	0.0	0.0	0.0	0.191847
2019-01-05	0.289298	-1.691300	1.977181	0.289298	0.289298	-0.124464	-0.124464	-0.124464	-0.124464	-0.124464	-0.124464	0.0	0.0	0.0	0.164834
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
2020-08-08	23.220561	21.173018	24.985330	23.217569	23.223635	-0.124464	-0.124464	-0.124464	-0.124464	-0.124464	-0.124464	0.0	0.0	0.0	23.096097
2020-08-09	23.260171	21.399980	25.122012	23.257116	23.263297	-0.034287	-0.034287	-0.034287	-0.034287	-0.034287	-0.034287	0.0	0.0	0.0	23.225884
2020-08-10	23.299781	21.463244	25.165483	23.296664	23.302973	0.050243	0.050243	0.050243	0.050243	0.050243	0.050243	0.0	0.0	0.0	23.350024
2020-08-11	23.339391	21.507816	25.258713	23.336213	23.342622	0.063973	0.063973	0.063973	0.063973	0.063973	0.063973	0.0	0.0	0.0	23.403364
2020-08-12	23.379001	21.503110	25.434978	23.375763	23.382289	0.083128	0.083128	0.083128	0.083128	0.083128	0.083128	0.0	0.0	0.0	23.462129
590 rows × 15 columns


# Produce a sales forecast for the finance division
# giving them a number for expected total sales next quarter.
# Provide best case (yhat_upper), worst case (yhat_lower), and most likely (yhat) scenarios.
​
# Create a forecast_quarter Dataframe for the period 2020-07-01 to 2020-09-30
# The DataFrame should include the columns yhat_upper, yhat_lower, and yhat
mercado_sales_forecast_quarter = mercado_sales_prophet_forecast.loc["2020-07-01":"2020-09-30"]
​
# Update the column names for the forecast_quarter DataFrame
# to match what the finance division is looking for 
mercado_sales_forecast_quarter = mercado_sales_forecast_quarter.rename(columns={
        "yhat_upper" : "Best Case",
        "yhat_lower" : "Worst Case", 
        "yhat" : "Most Likely Case"}
)
​
# Review the last five rows of the DataFrame
mercado_sales_forecast_quarter.tail()
trend	Worst Case	Best Case	trend_lower	trend_upper	additive_terms	additive_terms_lower	additive_terms_upper	weekly	weekly_lower	weekly_upper	multiplicative_terms	multiplicative_terms_lower	multiplicative_terms_upper	Most Likely Case
ds															
2020-08-08	23.220561	21.173018	24.985330	23.217569	23.223635	-0.124464	-0.124464	-0.124464	-0.124464	-0.124464	-0.124464	0.0	0.0	0.0	23.096097
2020-08-09	23.260171	21.399980	25.122012	23.257116	23.263297	-0.034287	-0.034287	-0.034287	-0.034287	-0.034287	-0.034287	0.0	0.0	0.0	23.225884
2020-08-10	23.299781	21.463244	25.165483	23.296664	23.302973	0.050243	0.050243	0.050243	0.050243	0.050243	0.050243	0.0	0.0	0.0	23.350024
2020-08-11	23.339391	21.507816	25.258713	23.336213	23.342622	0.063973	0.063973	0.063973	0.063973	0.063973	0.063973	0.0	0.0	0.0	23.403364
2020-08-12	23.379001	21.503110	25.434978	23.375763	23.382289	0.083128	0.083128	0.083128	0.083128	0.083128	0.083128	0.0	0.0	0.0	23.462129

# Displayed the summed values for all the rows in the forecast_quarter DataFrame
mercado_sales_forecast_quarter.sum()
trend                          969.529149
Worst Case                     888.283503
Best Case                     1051.081759
trend_lower                    969.435466
trend_upper                    969.620368
additive_terms                   0.083128
additive_terms_lower             0.083128
additive_terms_upper             0.083128
weekly                           0.083128
weekly_lower                     0.083128
weekly_upper                     0.083128
multiplicative_terms             0.000000
multiplicative_terms_lower       0.000000
multiplicative_terms_upper       0.000000
Most Likely Case               969.612277
dtype: float64
Based on the forecast information generated above, produce a sales forecast for the finance division, giving them a number for expected total sales next quarter. Include best and worst case scenarios, to better help the finance team plan.

Answer: Based on the forecast information data, the most likely sales number for next quarter will be approximately 969.5mm. Sales could exceed this if something closer to the best case scenario plays out. In the very best forecast case, sales would be approximately 1.051bb. Additionally, the worst forecast scenario would dip down toward 888mm in sales.


​

Simple
0
25
Python 3 (ipykernel) | Idle
1
forecasting_net_prophet.ipynb