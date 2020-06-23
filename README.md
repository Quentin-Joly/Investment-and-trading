# Investment-and-trading

## Project definition
Investment firms, hedge funds and even individuals have been using financial models to better understand market behavior and make profitable investments and trades. A wealth of information is available in the form of historical stock prices and company performance data, suitable for machine learning algorithms to process.

For this project, I have to to build a stock price predictor that takes daily trading data over a certain date range as input, and outputs projected estimates for given query dates. Note that the inputs contain multiple metrics, such as opening price (Open), highest price the stock traded at (High), how many stocks were traded (Volume) and closing price adjusted for stock splits and dividends (Adjusted Close); my system only needs to predict the Adjusted Close price.

For this project, i choose to create a web app to show the Adjusted Close price for each day with predicted prices for the future dates and a bar chart to show a daily, weekly or monthly growing rate.


## Analysis
The data is extracted from the Yahoo Finance site through an API
Input data:
- Date : the date of the saved data
- Open : the opening stock price 
- Low : the lowest stock price of the day
- High : the highest stock price of the day
- Close : the closing stock price of the day
- Adjusted Close : the Adjusted closing stock price 

Some Adjusted Close values are missing in the extracted data, so I filled them with the average between the previous and the next Adjusted Close value for visualizations purpose.


## User interface
By default, the application opens with the 'AAPL' ticker (for Apple Inc.) and without prediction. 
The user can choose one or various of any tickers, then choose a training period and a forecast date and click on "Predict" to show the graphs. *The training period selected must be superior to the predicted period*. 
The "Companies global growth" graph shows how much a company action grown up (positively or negatively) on the selected period. The user can select one or various tickers, select the growth period and click "Show" to display the graph.


## Methodology
### Data preprocessing
All prices are on the same unity and then, don't need to be modified and the date data is converted to seconds to datetime objects. I added a 'Company' and 'Data' columnsto identify where the daat come from and to know later if the daat is original or predicted.

### Implementation
#### Techniques
To make predictions, I use a bagging technique combined with Linear Regression and K Nearest Neighbors Regression models.
For low prediction date range I use the LR model to predict new values and for higher prediction range I use the KNN model.
I create bags of random data picked from the original data, train a model for each of those bags and then 
#### Algorithm
First I clean the data to remove the Nan values from the dataset. I from the original data, I keep 



