"""
Created on Sat Jul 16 15:50:37 2016

@author: nchandra
"""

import datetime
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, VarianceThreshold

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.decomposition import PCA

from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.grid_search import GridSearchCV


dir_loc = '..//input//'


def create_submission(y_hat):
    now = datetime.datetime.now()
    sub_pd = pd.read_csv(dir_loc + "sample_submission.csv")
    sub_file = 'submission_' + str(now.strftime("%Y-%m-%d_%H")+'Hr-'+now.strftime("%M")+'Min')
    col=sub_pd.columns.values
    sub_y_hat1 = pd.DataFrame()
    sub_y_hat1['device_id'] = sub_pd['device_id']
    sub_y_hat2 = pd.DataFrame(y_hat, columns=col[1:])
    sub_y_hat = pd.concat([sub_y_hat1, sub_y_hat2], axis=1)
    sub_y_hat.to_csv(dir_loc + sub_file + '.csv', sep=',', index=False)
#    print(sub_y_hat)
#    print(sub_pd.columns.values)
    return
    
def encode_onehot(df, cols):
    """
    One-hot encoding is applied to columns specified in a pandas DataFrame.
    
    Modified from: https://gist.github.com/kljensen/5452382
    
    Details:
    
    http://en.wikipedia.org/wiki/One-hot
    http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """
    vec = DictVectorizer()
    
    vec_data = pd.DataFrame(vec.fit_transform(df[cols].to_dict(orient='records')).toarray())
    vec_data.columns = vec.get_feature_names()
    vec_data.index = df.index
    
    df = df.drop(cols, axis=1)
    df = df.join(vec_data)
    return df

class dataset():
    '''
    # use this class to load data and do processing on it
    '''
    def __init__(self):
        self.train = []
        self.label = []
        self.test = []
        
    def __read_data(self):
        '''
        # load datasets
        '''
        print("Read datasets...")
        train = pd.read_csv(dir_loc + "gender_age_train.csv", dtype = {"device_id": np.str, "age": np.int8})
        test = pd.read_csv(dir_loc +  "gender_age_test.csv", dtype = {"device_id": np.str, "age": np.int8})
        events = pd.read_csv(dir_loc + "events.csv", dtype = {"device_id": np.str})
        app_events = pd.read_csv(dir_loc + "app_events.csv")
        app_labels = pd.read_csv(dir_loc+ "app_labels.csv")
        label_categories = pd.read_csv(dir_loc + "label_categories_v2.csv")
        brands = pd.read_csv(dir_loc + "phone_brand_device_model.csv", dtype = {"device_id": np.str}, encoding = "UTF-8")
        print("Done with datasets...")
        return train, test, events, app_events, app_labels, label_categories, brands
        
    def __code_testing(self):
        '''
        # Sample the data significantly to run code faster only for testing the code...
        '''
        train, test, events, app_events, app_labels, label_categories, brands = __read_data()
        train = train.sample(50)
        test = test.sample(50)
        events = events.sample(50)
        app_events = app_events.sample(50)
        return train, test, events, app_events, app_labels, label_categories, brands

    def __map_column(self, table, f):
        labels = sorted(table[f].unique())
        mappings = dict()
        for i in range(len(labels)):
            mappings[labels[i]] = i
        table = table.replace({f: mappings})
        return table

    def data_wrangling(self, code_testing=False):
        '''
        # do data_wrangling and return some transform version of train and test
        '''
        if code_testing:
            train, test, events, app_events, app_labels, label_categories, brands = __code_testing()
        else:
            train, test, events, app_events, app_labels, label_categories, brands = __read_data()
        
#        Merge label_categories with app_lebels
#        We'll avoice this for now as it is giving memory error
        app_lebels = pd.merge(app_lebels, label_categories, how='left', on='label_id')
        app_lebels = app_labels.drop(['category'], axis=1)
        
#        Merge App label and App events
        app_events = pd.merge(app_events, app_labels, how='left', on='app_id')
        
#        App events
        app_events['appcounts'] = app_events[['event_id','app_id']].groupby(['event_id'])['app_id'].transform('count')
        app_events['appcounts_isinst'] = app_events[['event_id','app_id','is_installed']][app_events.is_installed==1].groupby(['event_id'])['app_id'].transform('count')
        app_events['appcounts_isactv'] = app_events[['event_id','app_id','is_active']][app_events.is_active==1].groupby(['event_id'])['app_id'].transform('count')
        app_events['appcounts_isgame'] = app_events[['event_id','app_id','game']][app_events.game==1].groupby(['event_id'])['app_id'].transform('count')

        app_events_all = app_events[['event_id', 'appcounts']].drop_duplicates('event_id', keep='first')
        app_events_isinst = app_events[['event_id', 'appcounts_isinst']].drop_duplicates('event_id', keep='first')
        app_events_isavtv = app_events[['event_id', 'appcounts_isactv']].drop_duplicates('event_id', keep='first')
        app_events_isgame = app_events[['event_id', 'appcounts_isgame']].drop_duplicates('event_id', keep='first')

#        Now the tricky part
#        Drop the is_installed, is_active, and game
#        One hot encoding on label_id
#        groupby on events
        app_events = app_events.drop(['is_installed','is_active','game','appcounts','appcounts_isinst','appcounts_isactv','appcounts_isgame'], axis=1)
        app_events.label_id = app_events.label_id.astype('str')
        app_events = encode_onehot(app_events, cols=['label_id'])

        app_events['appcounts_label'] = app_events.groupby(['event_id'])['app_id'].transform('count')
        app_events_label = app_events[['event_id', 'appcounts_label']].drop_duplicates('event_id', keep='first')

#        Events
        events['counts'] = events.groupby(['device_id'])['event_id'].transform('count')
        events_small = events[['device_id', 'counts']].drop_duplicates('device_id', keep='first')
        
        e1=pd.merge(events, app_events_all, how='left', on='event_id', left_index=True)
        e1.loc[e1.isnull()['appcounts'] ==True, 'appcounts']=0
        e1['appcounts1'] = e1.groupby(['device_id'])['appcounts'].transform('sum')
        e1_small = e1[['device_id', 'appcounts1']].drop_duplicates('device_id', keep='first')

#        brands
        brands.drop_duplicates('device_id', keep='first', inplace=True)
        brands = __map_column(brands, 'phone_brand')
        brands = __map_column(brands, 'device_model')

#        train (Create lables and drop leakage variables)        
#        label
        self.label = train['group']
        train = train.drop(['age'], axis=1)
        train = train.drop(['gender'], axis=1)
        train = train.drop(['group'], axis=1)

#        train
        train = __map_column(train, 'group')
        train = pd.merge(train, brands, how='left', on='device_id', left_index=True)
        train = pd.merge(train, events_small, how='left', on='device_id', left_index=True)
        train = pd.merge(train, e1_small, how='left', on='device_id', left_index=True)
        
        self.train = train
        
#        test
        self.test = pd.merge(self.test, self.brands, how='left', on='device_id', left_index=True)
        self.test = pd.merge(self.test, events_small, how='left', on='device_id', left_index=True)
        self.test = pd.merge(self.test, e1_small, how='left', on='device_id', left_index=True)

#        print(self.train.head(5))
#        print(self.train.info())
#        print(self.train.columns.values)
#        print(self.test.columns.values)


        
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
#            print("Column(s) to select are {} :".format(self.col_names))
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
        print("Datatype for X is {}".format(type(X)))
        print("Dataframe shape for X is {}".format(X.shape))
        return X.fillna(-1)
#        return X.fillna(-1, inplace=True)
        
class InputerTransfrmer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        

class PrintTransfrmer(BaseEstimator, TransformerMixin):
    '''
    # simple class to validate the transformed data
    '''
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        print("Shape of X is: {}".format(X.shape))
        print("Type of X is: {}".format(type(X)))
#        print(X)
        return X

class GroupCntTransfrmer(BaseEstimator, TransformerMixin):
    '''
    # feature for count of group column
    '''
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return Counter(X)

class PCAonColTransfrmer(BaseEstimator, TransformerMixin):
    '''
    # feature for count of group column
    '''
    def __init__(self, column):
        self.column = column
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return PCA().fit_transform(X[self.column])

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
#                        PCA on specific 
                        ('pca_xform', Pipeline([
                            ('pca', PCA()),
                            ('selection', SelectKBest()),
                        ])),
                        ('', Pipeline([
                            ]))
                        
                        
                        
                        ]))
#                ('feature_union', FeatureUnion(
#                    transformer_list=[
##                         Select specific column
#                        ('cntr_device', Pipeline([
#                            ('print_data1', PrintTransfrmer()),
#                            ('group_col_selector', ColumnSelector(['device_id'])),
#                            ('cnt', GroupCntTransfrmer()),
#                            ])),
#                        ('cntr_brand', Pipeline([
#                            ('group_col_selector', ColumnSelector(['phone_brand'])),
#                            ('print_data2', PrintTransfrmer()),
#                            ('cnt', GroupCntTransfrmer()),
#                            ])),
#                    ],
#                    transformer_weights={
##                         weight components in FeatureUnion
#                        'cntr_device': 1.0,
#                        'cntr_brand': 1.0,
#                    },
#                    )
#                ),
                ('print_data', PrintTransfrmer()),
                ('estimators', FeatureUnion([
                    ('gbc', GradientBoostingClassifier()),
                    ('rf', RandomForestClassifier()),
                    ])
                ),
                ('ensambler', LogisticRegression()),
                ])

pipe_params = {#'feature_union__transformer_weights':[[1,1], [4,1], [1,4]],
               'estimators__gbc__n_estimators': [100, 500, 1500],
               'estimators__rf__n_estimators': [100, 500, 1500],
               'ensambler__C': [10, 1, 0.1],
                }

if __name__=='__main__':
    all_data = dataset()
    all_data.data_wrangling(code_testing=False)
    print('Performing Grid Search...')
    grid_search = GridSearchCV(pipe, pipe_params, cv=2, scoring='log_loss', verbose=3)    
    grid_search.fit(all_data.train, all_data.label)
    
    print("Best Score {}".format(grid_search.best_score_))
    print("Best Model...:")
    print(grid_search.best_estimator_.get_params())
    
    y_hat = grid_search.predict_proba(all_data.test)
#    y_hat = grid_search.best_estimator_.predict_prob(all_data.test)
    print("Printing the predictions...")
    print(type(y_hat))
    #print(y_hat)
    create_submission(y_hat)










