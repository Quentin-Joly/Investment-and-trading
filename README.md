# Investment-and-trading

## Installation
 - Pandas
 - Numpy
 - Requests
 - json
 - Plotly
 - Flask
 - Datetime
 - sklearn


## How to run
 - Run the 'run.py' file in your terminal
 - Open the 'http://127.0.0.1:5000' url in your navigator


## File explanation
 - run.py : the launcher of the application
 - predicator.py : the ML pipeline I used to train the models and retrun a bag of models and the associate score
 - master.html : the HTML template used for visualizations


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
The user can choose one or various of any tickers, then choose a training period and a forecast date and click on "Predict" to show the graphs. **The training period selected must be superior to the predicted period**. 
The "Companies global growth" graph shows how much a company action grown up (positively or negatively) on the selected period. The user can select one or various tickers, select the growth period and click "Show" to display the graph.


## Methodology
### Data preprocessing
All prices are on the same unity and then, don't need to be modified and the date data is converted to seconds to datetime objects. I added a 'Company' and 'Data' columns to identify where the daat come from and to know later if the daat is original or predicted.

### Implementation
#### Techniques
To make predictions, I use a bagging technique combined with Linear Regression and K Nearest Neighbors Regression models.
For low prediction date range I use the LR model to predict new values and for higher prediction range I use the KNN model.
I create bags of random data picked from the original data and train a model for each of those bags. Then I compare the average predictions made by each bags to the average test data to compute a R² score. Finally I use these bags to predict the desired data wished by the user and take the average result to have a better prediction.
#### Algorithm
1. I request the tickers selected by the user, if none have been selected, the default value is 'AAPL'
2. I load the last year data from the Yahoo Finance site through an API with the tickers selected. If various tickers have been selected, all data from each tickers is appended in a single dataframe
3. Some 'Adjusted Close' data are missing so i fill them by taking the average between the previous and the next 'Adjusted Close' value for visualization purpose.
4. I request the train duration and the forecast wished by the user
5. For each ticker selected by the user :
    - If the train duration and the forecast have been selected :
    1. For each day up to the predicted required date, I predict the future value (ex. for a 3 days prediction, I make a prediction for the 1st day, for the 2nd day and for the 3rd day independently). I create as well the worst case scenarios by computing the lowest and the highest prediction possible based on the prediction and on the score (ex. prediction = value, 0 < score < 1, lowest prediction = value * score, highest prediction = value / score)
    2. I append the result of each day in a dataframe
    3. I append the last date value of the original data inside the prediction dataframes for visualization purpose
    - If the train duration or the forecast haven't been selected, I leave the prediction dataframes empty
    - I append the predicted values for a ticker in a dictionnary which matches tickers to its predictions
6. I request the growth period selected by the user, If none have been selected, the default value is 1 day
7. I compute the growth for the selected period
8. I generate data for the graphs with the original data and the predicted data
9. I set the graphs data and the layout
10. I export the graphs objects to the HTML template
#### Metrics
For those models, I use the R² score metric to evaluate the similarity of the predictions with the test values.

### Refinelent 
First I only tried a LR model to try to predict the adjusted close values but the scores and predictions were too far from the test values. I improved the model by adding the KNN model and combined these 2 models with the bagging method to optimize the predictions with the LR model performing better with a low perdiction date and the KNN model perfoming better with a high predicted date.


## Results
As I tried to maximize the prediction scores, I compute the daily prediction and append them in a dataframe, so I have a different score for each day. In theory, the closer the prediction date is from the original data, the better the score is but in reality, the score can vary a lot and can go down to 0.2. I don't excatly know why the score vary from 0.8 to 0.2 the next day, but I guess it can be because of the variability of the original data. The score already have been imporove a lot by combining the techniques and the models, but it still could be improved.


## Conclusion
The difficulties of this project to me is to find a balance between the models used and the use of the bagging technique.
This project could be improved by training a model to sell, hold or buy stocks to make a profit.
I guess the predictions could be improved as well.
