import pickle
import inflection
import numpy as np
import pandas as pd
from datetime import datetime


class StoreSales(object):
    
    def __init__(self):
        
        self.home_path=''
        self.competition_distance_scaler   = pickle.load(open(self.home_path + 'rescaling/competition_distance_scaler.pkl', 'rb'))
        self.competition_months_old_scaler = pickle.load(open(self.home_path + 'rescaling/competition_months_old_scaler.pkl', 'rb'))
        self.promo2_months_old_scaler      = pickle.load(open(self.home_path + 'rescaling/promo2_months_old_scaler.pkl', 'rb'))
        self.store_type_scaler             = pickle.load(open(self.home_path + 'rescaling/store_type_scaler.pkl', 'rb'))
        self.year_scaler                   = pickle.load(open(self.home_path + 'rescaling/year_scaler.pkl', 'rb'))
    
    
    def data_cleaning(self, df1):
        
        # renaming column names
        snake_case = lambda x: inflection.underscore(x)
        df1.columns = list(map(snake_case, df1.columns))

        # setting date column as datetime type
        df1['date'] = pd.to_datetime(df1['date'])

        # Filling in Missing/Null Values

        # competition_distance - using maximum distance x 2
        # maximun distance x 2
        max_dist_x_2 = df1['competition_distance'].max() * 2
        # assuming competitors are twice as far away as the greatest distance found
        df1['competition_distance'] = df1['competition_distance'].apply(lambda x: max_dist_x_2 if np.isnan(x) else x)

        # competition_open_since_year
        # latest year
        df1.loc[df1['competition_open_since_year'].isna(), 'competition_open_since_year'] = df1['date'].max().year

        # competition_open_since_month
        # latest month
        df1.loc[df1['competition_open_since_month'].isna(), 'competition_open_since_month'] = df1['date'].max().month

        # promo2_since_week AND promo2_since_year

        # in case of NA values the date of sale will be used
        # the difference between these dates will be used later
        #promo2_since_week
        df1['promo2_since_week'] = df1[['date', 'promo2_since_week']].apply(lambda x: x['date'].week if np.isnan(x['promo2_since_week']) else x['promo2_since_week'], axis=1)
        # promo2_since_year
        df1['promo2_since_year'] = df1[['date', 'promo2_since_year']].apply(lambda x: x['date'].year if np.isnan(x['promo2_since_year']) else x['promo2_since_year'], axis=1)


        # Changing Data Types

        # Changing DTypes from float to integer
        df1['competition_distance'] = df1['competition_distance'].astype(int)
        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype(int)
        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype(int)
        df1['promo2_since_week'] = df1['promo2_since_week'].astype(int)
        df1['promo2_since_year'] = df1['promo2_since_year'].astype(int)

        return df1


    def feature_engineering(self, df2):

        # date related features
        # year
        df2['year'] = df2['date'].dt.year
        # month
        df2['month'] = df2['date'].dt.month
        # day
        df2['day'] = df2['date'].dt.day
        # week_of_year
        df2['week_of_year'] = df2['date'].dt.isocalendar().week.astype('int64')

        # competition_months_old
        # calculating the competition period, extracting the days and dividing by 30 to get the period in months
        df2['competition_months_old'] = df2.apply(lambda x: (
            x['date'] - datetime(year=x['competition_open_since_year'], 
                                 month=x['competition_open_since_month'], 
                                 day=1)).days / 30, axis=1).astype(int)
        # assigning zero to negative values of competition_months_old
        # in this case it makes no sense to work with the time that is left for the competitor to open
        df2.loc[df2['competition_months_old'] < 0, 'competition_months_old'] = 0


        # promo2_months_old
        # calculation method: zero(0) if promo2 is zero(0) else (actual_date - promo2_starting_date) >> timedelta format 
        # >> then use .days and divide by 30 to extract the number of months >> as integer
        df2['promo2_months_old'] = df2.apply(lambda x: 0 if x['promo2'] == 0 else (
            x['date'] - datetime.fromisocalendar(x['promo2_since_year'], 
                                                 x['promo2_since_week'], 
                                                 1)).days / 30, axis=1).astype(int)
        # assigning zero to negative values of promo2_months_old
        # since the store is not yet participating (but will in the future)
        df2.loc[df2['promo2_months_old'] < 0, 'promo2_months_old'] = 0


        # Filtering Features

        # eliminating all records where stores are closed 
        df2 = df2[(df2['open'] != 0) & (df2['open'].notna())]

        # columns to be droped
        cols_drop = ['id', 'open', 'promo_interval']
        df2.drop(cols_drop, axis=1, inplace=True)
    
        return df2
    
    
    def data_preparation(self, df5):
        
        # Rescaling

        # Robust Scaler
        # competition_distance
        df5['competition_distance'] = self.competition_distance_scaler.fit_transform(df5[['competition_distance']].values)
        # competition_months_old
        df5['competition_months_old'] = self.competition_months_old_scaler.fit_transform(df5[['competition_months_old']].values)

        # Min-Max Scaler
        # promo2_months_old
        df5['promo2_months_old'] = self.promo2_months_old_scaler.fit_transform(df5[['promo2_months_old']].values)
        # year
        df5['year'] = self.year_scaler.fit_transform(df5[['year']].values)


        # Feature Transformation

        # Enconding: Transforming Categorical Features Into Numeric Features
        # assortment - Ordinal Encoding
        assortment_dict = {'a': 1, 'b': 2, 'c': 3}
        df5['assortment'] = df5['assortment'].map(assortment_dict)
        # store_type - Label Encoding
        df5['store_type'] = self.store_type_scaler.fit_transform(df5['store_type'])
        
        # Nature Transformation 
        # month
        df5['month_sin'] = df5['month'].apply(lambda x: np.sin(x * (2. * np.pi / 12)))
        df5['month_cos'] = df5['month'].apply(lambda x: np.cos(x * (2. * np.pi / 12)))
        # day
        df5['day_sin'] = df5['day'].apply(lambda x: np.sin(x * (2. * np.pi / 30)))
        df5['day_cos'] = df5['day'].apply(lambda x: np.cos(x * (2. * np.pi / 30)))
        # day_of_week
        df5['day_of_week_sin'] = df5['day_of_week'].apply(lambda x: np.sin(x * (2. * np.pi / 7)))
        df5['day_of_week_cos'] = df5['day_of_week'].apply(lambda x: np.cos(x * (2. * np.pi / 7)))
        # week_of_year
        df5['week_of_year_sin'] = df5['week_of_year'].apply(lambda x: np.sin(x * (2. * np.pi / 52)))
        df5['week_of_year_cos'] = df5['week_of_year'].apply(lambda x: np.cos(x * (2. * np.pi / 52)))
        
        # Boruta Selected Features
        selected_features = ['store', 'promo', 'store_type', 'assortment', 'competition_distance', 'competition_open_since_month', 
                             'competition_open_since_year', 'promo2', 'promo2_since_week', 'promo2_since_year', 'competition_months_old', 
                             'promo2_months_old', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'day_of_week_sin', 'day_of_week_cos', 
                             'week_of_year_sin', 'week_of_year_cos']

        return df5[selected_features]
    
    
    def get_prediction(self, model, original_data, test_data):
        
        # prediction
        predictions = model.predict(test_data)
        
        # join original_data and predictions
        original_data['prediction'] = np.expm1(predictions)
        
        return original_data.to_json(orient='records', date_format='iso')
