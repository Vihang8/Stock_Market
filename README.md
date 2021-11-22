# Stock_Market
A practical study to understand performance learn-
ing and its application for predicting closing stock prices by calculating unknown
weights in an Auto-regression model.

an exposure to the implementation of LMS algorithm,
the effects that hyperparameters like number of epochs and learning rate have
on the convergence and accuracy of the algorithm and finally to determine
the reliability of the predictor and conclude the issues that might be possibly
present.

The data under consideration is of the daily closing prices of stock index of
Apple (https://finance.yahoo.com/quote/AAPL/history/). There are in total 1004 data points under-consideration.
The closing prices of first 336 days form the training set, the closing prices of
the next 336 form validation set and the closing prices of remaining days form
the testing set.

Finally, the data is normalized to contain the growth of weights. It was ob-
served that the weight vector displayed [nannannan] if the stock prices were
considered in their dollar value. To maintain the integrity of the testing and
validation data, the entire dataset was normalized by considering the mean and
variance of the training data only.

Model : 
The data is considered as time-series and thus is modeled using Auto-Regression
as following.

y(n) = w[0] ∗ x[n − 1] + w[1] ∗ x[n − 2] + w[2] ∗ x[n − 3] + noise[n]

The closing stock price for day n is predicted by considering the closing
prices on days (n-1), (n-2) and (n-3) and finding a linear predictor which maps
the input to the output.

The noise vector is constructed by assuming the noise to be zero mean white
Gaussian random process with variance the same as data and the number of
points same as data.
