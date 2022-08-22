# StockExchangePrediction
Full name of my Banchelor thesis: Stock Exchange Prediction Machine Learning Algorithm / (RO)Tehnici de invatare automata pentru predictia indiciilor bursieri.
This app was made for my banchelor's thesis degree. It's in romanian language, so it might not make any sense for english speakers.
The predictions are made using ML scripts for regression. Forecasting techniques used are: Linear Regression, KNN, AUTO-ARIMA and LSTM.
This app is made to predict the next day of the current/selected .csv price history using the closing price or 'Close' column.

NOTE: The app is made for scientific research for my Bachelor's Thesis, so DO NOT EXPECT REAL PREDICTIONS.(even that lstm had some good results)

BEFORE RUN:
In app/routes.py at lines 165-169 are some hardcoded paths. Please change it up with your path where the project at. 
For some reasons i was unable to make it better. I know it's bad practice, but the project was made only for my setup.

To run this app:
1. Install requirements.txt in (new) virtual env.
2. Open terminal into the project folder and type 'flask run'
3. Done!

How to use:
1. First, select one or more(or all) of the four algorithms.
2. Select the dataset. You have 2 options: Specifying the company stock symbol or manual upload( MUST BE 1260 LINES IN .CSV OR LAST 5 YEARS HISTORY, AND THE CLOSE PRICE COLUMN MUST BE NAMED 'Close').
3. Wait and then see the results!


BUGS KNOWN:
I don't know why but sometimes(and especially on other computers) the program cannot read the numpy module, even that it's installed and it's working well, so it might crash. You could run the desired ML algotihm script solo and it will work well.
