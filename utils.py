import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose


class WeekendFillers:
    def __init__(self, data:pd.DataFrame):
        self.df = data

    def fill_weekend_data(self): # using mean between week days
        # Set the date column as the index
        df = self.df.set_index('date')

        # Create a new DataFrame with all days included
        full_df = pd.DataFrame(index=pd.date_range(df.index.min(), df.index.max()))

        # Join the full DataFrame with the original DataFrame
        full_df = full_df.join(df)

        # Forward fill missing values
        full_df = full_df.fillna(method='ffill')

        # Backward fill remaining missing values
        full_df = full_df.fillna(method='bfill')

        # Calculate the means for each weekend date
        means = (full_df.loc[full_df.index.weekday == 0]['close'] + full_df.loc[full_df.index.weekday == 4]['close']) / 2

        # Set the means for the missing weekend dates
        full_df.loc[full_df.index.weekday == 5, 'close'] = means
        full_df.loc[full_df.index.weekday == 6, 'close'] = means

        # Reset the index and return the filled DataFrame
        full_df = full_df.reset_index().rename(columns={'index': 'date'})

        return full_df

    def fill_weekend_gaps(self): # gap filling over seasonal decomposition
        # Resample the DataFrame to have a row for every day
        daily_df = self.df.resample('D').asfreq()

        # Find the weekends
        weekends = daily_df[daily_df.index.weekday >= 5].index

        # Perform seasonal decomposition on the time series
        result = seasonal_decompose(daily_df, model='additive')

        # Fill in the gaps with the seasonal component
        for weekend in weekends:
            year = weekend.year
            month = weekend.month
            day = weekend.day
            avg_seasonal = result.seasonal[(result.seasonal.index.month == month) &
                                           (result.seasonal.index.day == day)].mean()
            daily_df.loc[weekend] = result.trend.loc[weekend] + avg_seasonal

        return daily_df