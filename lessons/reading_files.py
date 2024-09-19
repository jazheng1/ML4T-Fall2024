import pandas as pd
import matplotlib.pyplot as plt

def test_run():
    """Function called by Test Run."""
    df = pd.read_csv("../data/AAPL.csv")
    # Quiz: Print last 5 rows of the data frame
    # print df				# prints entire data set (dataframe)
    # print df.head()		# prints first five records
    #print(df.tail())  # prints last five records
    print(df[10:21])  # print rows between index 10 and 20 inclusive


"""Plotting Stock Price Data"""
def plot_run():
    """Plot a single column."""
    df = pd.read_csv("../data/AAPL.csv")
    print (df['Adj Close'])
    df[['Close', 'Adj Close']].plot()
    plt.show()  # must be called to show plots


if __name__ == "__main__":
    plot_run()