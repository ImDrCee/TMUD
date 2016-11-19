"""
Created on Thu Jul 14 08:48:05 2016

@author: nchandra
"""

import datetime
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

dir_loc = '..//input//'


def create_submission(y_hat):
    now = datetime.datetime.now()
    sub_pd = pd.read_csv(dir_loc + "sample_submission.csv")
    sub_file = 'submission_' + str(now.strftime("%Y-%m-%d_%H")+'Hr-'+now.strftime("%M")+'Min') + '.csv'
    col=sub_pd.columns.values
    sub_y_hat1 = pd.DataFrame()
    sub_y_hat1['device_id'] = sub_pd['device_id']
    sub_y_hat2 = pd.DataFrame(y_hat, columns=col[1:])
    sub_y_hat = pd.concat([sub_y_hat1, sub_y_hat2], axis=1)
    sub_y_hat.to_csv(dir_loc + sub_file, sep=',', index=False, compression='zip')
#    print(sub_y_hat)
#    print(sub_pd.columns.values)
    return
    

class dataset():
    '''
    # use this class to load data and do processing on it
    '''
    def __init__(self):
        '''
        # load datasets
        '''
        print("Read datasets...")
        self.train = pd.read_csv(dir_loc + "gender_age_train.csv", dtype = {"device_id": np.str, "age": np.int8})
        self.test = pd.read_csv(dir_loc +  "gender_age_test.csv", dtype = {"device_id": np.str, "age": np.int8})
        self.events = pd.read_csv(dir_loc + "events.csv", dtype = {"device_id": np.str})
        self.brands = pd.read_csv(dir_loc + "phone_brand_device_model.csv", dtype = {"device_id": np.str}, encoding = "UTF-8")
        self.app_events = pd.read_csv(dir_loc + "app_events.csv")
        self.app_labels = pd.read_csv(dir_loc+ "app_labels.csv")
        self.label_categories = pd.read_csv(dir_loc + "label_categories.csv")
        print("Done with datasets...")
        
    def code_testing(self):
        '''
        # Sample the data significantly to run code faster only for testing the code...
        '''
        self.train = self.train.sample(50)
        self.test = self.test.sample(50)
        self.events = self.events.sample(50)
        self.app_events = self.app_events.sample(50)
                
    def data_wrangling(self):
        '''
        # do data_wrangling and return some transform version of train and test
        '''
        self.events['counts'] = self.events.groupby(['device_id'])['event_id'].transform('count')
        self.events_small = self.events[['device_id', 'counts']].drop_duplicates('device_id', keep='first')
        
        self.brands.drop_duplicates('device_id', keep='first', inplace=True)
        self.brands = self.map_column(self.brands, 'phone_brand')
        self.brands = self.map_column(self.brands, 'device_model')
        
        self.train = self.map_column(self.train, 'group')
        self.train = self.train.drop(['age'], axis=1)
        self.train = self.train.drop(['gender'], axis=1)
        self.train = pd.merge(self.train, self.brands, how='left', on='device_id', left_index=True)
        self.train = pd.merge(self.train, self.events_small, how='left', on='device_id', left_index=True)
        self.label = self.train['group']
        self.train = self.train.drop(['group'], axis=1)
#        self.train.fillna(-1, inplace=True)
        
        self.test = pd.merge(self.test, self.brands, how='left', on='device_id', left_index=True)
        self.test = pd.merge(self.test, self.events_small, how='left', on='device_id', left_index=True)
#        self.test.fillna(-1, inplace=True)
#        print(self.train.head(5))
#        print(self.train.columns.values)
#        print(self.test.columns.values)


    def map_column(self, table, f):
        labels = sorted(table[f].unique())
        mappings = dict()
        for i in range(len(labels)):
            mappings[labels[i]] = i
        table = table.replace({f: mappings})
        return table
        
    def data_wrangling_v2(self, TorT):
        '''
        # Take data after loading and merge in appropriate ways...
        # Merging datasets and dropping unnecessary items
        '''
        print(TorT)
        print('.....')
        ### events
        self.TorT = self.TorT.merge(self.events, how = "left", on = "device_id")
        # del self.events
        
        ### brands
        self.TorT = self.TorT.merge(self.brands, how = "left", on = "device_id")
        # del self.brands
        self.TorT.drop("device_id", axis = 1, inplace = True)
        # self.test.drop("device_id", axis = 1, inplace = True)
        
        ### app_events
        self.TorT = self.TorT.merge(self.app_events, how = "left", on = "event_id")
        # del self.app_events
        self.TorT.drop("event_id", axis = 1, inplace = True)
        
        ### app_labels and label_categories
        self.TorT = self.TorT.merge(self.app_labels, how = "left", on = "app_id")
        self.TorT = self.TorT.merge(self.label_categories, how = "left", on = "label_id")
        # del self.app_labels, self.label_categories
        self.TorT.drop(["app_id", "label_id"], axis = 1, inplace = True)
        
        print(self.TorT.head(5))
        print(self.TorT.info())
        
class ColumnSelector(BaseEstimator, TransformerMixin):
    '''
    # Selected specific columns from the given data frame
    # If no column names are provided for selection the entire dataframe is returned
    '''
    def __init__(self, col_names=[]):
        self.col_names = col_names

    def fit(self, x, y=None):
        return self

    def transform(self, data_frame):
        if self.col_names == []:
            return data_frame
        else:
#            print("Column(s) to select are {} :", self.col_names)
#            print(data_frame)
            return data_frame[self.col_names]

class DataTransfrmer(BaseEstimator, TransformerMixin):
    '''
    # do something for the entire feature set
    # like extract subject and body or impute empty cells
    # or something else...
    # write custom transformer for it
    '''    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
#        print("length of dataframe: {}", len(X))
        Xt = X.fillna(-1)
#        print(type(Xt), type(X))
#        print(Xt)
#        return X.fillna(-1, inplace=True)
        return Xt

class PrintTransfrmer(BaseEstimator, TransformerMixin):
    '''
    # simple class to validate the transformed data
    '''
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        print(len(X))
        print(type(X))
#        print(X)
        return X

# Create the feature pipeline along with transformer_weights
pipe = Pipeline([
                # do something for the entire feature set
                # like extract subject and body or impute empty cells
                # or something else...
                # write custom transformer for it
                ('data_transformer', DataTransfrmer()),
                # feature union to select the feature from subsets
                ('feature_union', FeatureUnion(
                    transformer_list=[
#                         Select specific column
                        ('device_col_selector', ColumnSelector(['device_id', 'phone_brand', 'device_model'])),
                        ('brand_col_selector', ColumnSelector(['phone_brand'])),
                    ],
                    transformer_weights={
#                         weight components in FeatureUnion
                        'group_col_selector': 1.0,
                        'brand_col_selector': 1.0,
                    },
                    )
                ),
                ('print_data', PrintTransfrmer()),
                ('estimators', FeatureUnion([
                    ('svc', LinearSVC(multi_class='ovr')),
                    ('rf', RandomForestClassifier())
                    ])
                ),
                ('ensambler', LogisticRegression()),
                ])

all_data = dataset()
#all_data.code_testing()
all_data.data_wrangling()
pipe.fit(all_data.train, all_data.label)
y_hat = pipe.predict_proba(all_data.test)
print("Printing the predictions...")
print(type(y_hat))
#print(y_hat)
create_submission(y_hat)






