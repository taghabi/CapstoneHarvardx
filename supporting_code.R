# Install missing packages if not yet installed.
if(!require(lubridate)) install.packages("lubridate")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(prophet)) install.packages("prophet")
# Load Lubridate for data wrangling
library(lubridate)
# Load Ggplot for data visualization
library(ggplot2)
# Load Tidyverse for data manipulation
library(tidyverse)
# Load Prophet for data forecasting
library(prophet)

# Load the datasets from csv files
bitcoin <- read.csv("BTC-USD.csv", header = TRUE)
ethereum <- read.csv("ETH-USD.csv", header = TRUE)
# Both csv files were retrieved on 01/07/2020, directly from Yahoo finance.

# Preview the Bitcoin and Ethereum datasets
head(bitcoin)
head(ethereum)
# Preview the structure
str(bitcoin)
# There are 2305 observations and 7 variables
str(ethereum)
# There are 1981 observations and 7 variables.

# DATA WRANGLING
# In this time-series forecasting analysis, the main variables we will be using are Date and Closing Price. 
# Later on we will be using all historical data observations to develop a prediction model for Crypto prices in the next 365 days.
# To ensure the Prophet function has proper functionality, the Date and Closing price classes need to be in corrected for both datasets.
# Update Date class character to  Date
class(bitcoin$Date)
bitcoin$Date <- as.Date(bitcoin$Date)
class(ethereum$Date)
ethereum$Date <- as.Date(ethereum$Date)

# Update Close class from character to Numeric
class(bitcoin$Close)
bitcoin$Close <- suppressWarnings(as.numeric(as.character(bitcoin$Close)))
class(ethereum$Close)
ethereum$Close <- suppressWarnings(as.numeric(as.character(ethereum$Close)))

# DATA VISULIZATION
# Plot Bitcoin and Ethereum
suppressWarnings(qplot(Date, Close, data = bitcoin, main = 'Bitcoin closing prices 2014-2021'))
suppressWarnings(qplot(Date, Close, data = ethereum, main = 'Ethereum closing prices 2015-2021'))

# The closing prices will be transformed using log transformation to enable easier data manipulation and forecasting later on.
# Log transformation Bitcoin
ds <- bitcoin$Date
y <- log(bitcoin$Close)
df1 <- data.frame(ds, y)
qplot(ds, y, data = df1, main = 'Bitcoin closing in log scale')

# Log transformation Ethereum
ds <- ethereum$Date
y <- log(ethereum$Close)
df2 <- data.frame(ds, y)
qplot(ds, y, data = df2, main = 'Ethereum closing prices log scale')

# Note that the prophet function naming conventions are as follows
# ds for date, and y for closing price

# DATA FORECASTING - BITCOIN
# Calling the Prophet Function to Fit the Bitcoin Model
ds <- bitcoin$Date
y <- log(bitcoin$Close)
model1 <- prophet(df1)
future_price1 <- make_future_dataframe(model1, periods = 365)
# View last few values of the Future Price 1 data frame
tail(future_price1)
# We can validate that the predictions are made until January 2022, 1 year from now.
forecast1 <- predict(model1, future_price1)
tail(forecast1[c('ds','yhat','yhat_lower','yhat_upper')])

# PLOTING THE MODELS - BITCOIN 
plot(model1, forecast1)
# The black dots are the actual data points.
# The dark blue line is the predicted close estimate
# The light blue represents the confidence interval, with more time, the confidence expands wider due to uncertainty

# Interactive plot that provides date, actual and predicted values
dyplot.prophet(model1, forecast1)

# As seen in both plots, the price of Bitcoin begins to surge away from the predicted line at the start of 2021
# The new Bitcoin value reach can be calculated using the exponent function to transform back from log scale
# On January 5th, the Actual price was 
exp(10.43)
# While the predicted value was
exp(9.78)
# a difference of
exp(10.43) - exp(9.78)

prophet_plot_components(model1, forecast1)
# As seen in the first graph, Bitcoin will continue its increasing trend from 2020 into 2022.
# As seen in the second graph, Bitcoin prices are at usually their highest during the start of the working week, and usually at their lowest towards the end of the working week.
# Looking at the yearly pattern, closing prices are at their highest at January. 

# DATA FORECASTING - ETHEREUM
# Calling the Prophet Function to Fit the Ethereum Model
ds <- ethereum$Date
y <- log(ethereum$Close)
model2 <- prophet(df2)
future_price2 <- make_future_dataframe(model2, periods = 365)
# View last few values of the Future Price 2 data frame
tail(future_price2)
# Again, we can validate that the predictions are made until January 2022, 1 year from now.
forecast2 <- predict(model2, future_price2)
tail(forecast2[c('ds','yhat','yhat_lower','yhat_upper')])

# PLOTING THE MODELS - ETHEREUM 
plot(model2, forecast2)
# The black dots are the actual data points.
# The dark blue line is the predicted close estimate
# In the case of Ethereum, the confidence interval limits are wider due to higher uncertainty, but the trendline still points to an increase in closing price

# Interactive plot that provides date, actual and predicted values
dyplot.prophet(model2, forecast2)
# Again with Ethereum, the price begins to sure at the start of 2021
# on January 5th, the Actual price was
exp(7)
# While the predicted value was
exp(6.3)
# a difference of 
exp(7) - exp(6.3)

prophet_plot_components(model2, forecast2)
# Similar to Bitcoin, will continue its increasing trend from 2020 into 2022.
# Historically, Ethereum closing prices peak on Fridays and Wednesdays, and are at their lowest on Mondays
# Throughout a Historical year, Ethereum prices Peak at June and are at their lowest around May


# RESULTS - BITCOIN
prediction1 <- forecast1$yhat[1:2304]
actual1 <- model1$history$y
plot(actual1, prediction1)
abline(lm(prediction1 ~ actual1), col = 'red')
# The predicted points fall close to the predicted line with some variability
summary(lm(prediction1 ~ actual1))
# R-Squared is 0.9932, so about 99% of this model is explained by variability
x1 <- cross_validation(model1, 365, units = "days")
performance_metrics(x1, rolling_window = 0.1)
plot_cross_validation_metric(x1, metric = 'rmse', rolling_window = 0.1)
# root mean square error is represented in the grey plots 
# as shown in the graph, the rolling window line moves along with the rmse


# RESULTS - ETHEREUM
prediction2 <- forecast2$yhat[1:1980]
actual2 <- model2$history$y
plot(actual2, prediction2)
abline(lm(prediction2 ~ actual2), col = 'red')
# The predicted points fall close to the predicted line with some variability
summary(lm(prediction2 ~ actual2))
# R-Squared is 0.989, so about 98% of this model is explained by variability
x2 <- cross_validation(model2, 365, units = "days")
performance_metrics(x2, rolling_window = 0.1)
plot_cross_validation_metric(x2, metric = 'rmse', rolling_window = 0.1)
# root mean square error is represented in the grey plots 
# as shown in the graph, the rolling window line moves along with the rmse

# In conclusion...
# This model started off with predicting the future prices of cryptocurrency and was inspired from the recent boom surge
# to further strengthen this model, an additional factor of considering how many shares are bought (share buying activity) and to deteremine a relationship between price and buying and selling activity, to include the surge of buying 