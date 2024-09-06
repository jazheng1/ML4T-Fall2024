"""Build a DataFrame in Pandas"""

import pandas as pd


def test_run():
    # Define data range
    start_date = '2010-01-22'
    end_date = '2010-01-26'
    dates = pd.date_range(start_date, end_date)

    #print(dates)
    #print(dates[0])  # get first element of list

    # Create an empty dataframe
    df1 = pd.DataFrame(index=dates)  # define empty dataframe with these dates as index, or else it will be 0,1,2,3

    #print(df1)

    # Read SPY data into temporary dataframe
    # dfSPY = pd.read_csv("data/SPY.csv") # will result in no data because this has index of integers
    # dfSPY = pd.read_csv("data/SPY.csv", index_col="Date", parse_dates=True)
    dfSPY = pd.read_csv("../data/SPY.csv", index_col="Date",
                        parse_dates=True, usecols=['Date', 'Adj Close'],
                        na_values=['nan'])
    dfSPY = dfSPY.rename(columns={'Adj Close': 'SPY'})
    # print(dfSPY)

    # Join the two dataframes using DataFram.join()
    df1 = df1.join(dfSPY, how="inner")
    # print(df1)

    # Drop NaN Values
    # df1 = df1.dropna()
    # print(df1)

    symbols = ['GOOG', 'IBM', 'GLD']
    for sym in symbols:
        df_temp = pd.read_csv("../data/{}.csv".format(sym), index_col="Date",
                        parse_dates=True, usecols=['Date', 'Adj Close'],
                        na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': sym})
        df1 = df1.join(df_temp, how="inner")
        
    print(df1)
if __name__ == "__main__":
    test_run()