"""
Created on Sun Jul 24 20:38:01 2016

@author: nchandra
"""

"""
v2
Will not compbine all columns in one feature.
Combine feature one at a time.
Build and test classifiers for each feature combination.
Finally combine all classifiers in a pipeline
Read each file into the object and then create features as required.
"""
"""
v3
combine new features
"""

import datetime
import numpy as np
import pandas as pd
#from collections import Counter
from sklearn.preprocessing import LabelEncoder
#from sklearn.feature_extraction import DictVectorizer
#from sklearn.feature_selection import SelectKBest, VarianceThreshold

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion

#from sklearn.decomposition import PCA

from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.naive_bayes import GaussianNB
from sklearn.grid_search import GridSearchCV


dir_loc = '..//input//'


def create_submission(device_id, y_hat):
    now = datetime.datetime.now()
    sub_pd = pd.read_csv(dir_loc + "sample_submission.csv", dtype = {"device_id": np.str, "age": np.int8})
    sub_file = 'submission_' + str(now.strftime("%Y-%m-%d_%H")+'Hr-'+now.strftime("%M")+'Min')
    col=sub_pd.columns.values
    sub_y_hat1 = pd.DataFrame()
    sub_y_hat1[col[0]] = device_id
    sub_y_hat2 = pd.DataFrame(y_hat, columns=col[1:])
    sub_y_hat = pd.concat([sub_y_hat1, sub_y_hat2], axis=1)
    sub_y_hat.to_csv(dir_loc + sub_file + '.csv', sep=',', index=False)

class dataset():
    '''
    # use this class to load data and do processing on it
    '''
#    def __init__(self):
#        self.train = []
#        self.label = []
#        self.test = []
#        self.test_device_id = []
        
    def __read_data_tt(self):
        '''
        # load datasets
        '''
        print("Read datasets...")
        train = pd.read_csv(dir_loc + "gender_age_train.csv", dtype = {"device_id": np.str, "age": np.int8})
        self.train = pd.DataFrame(train['device_id'])
        test = pd.read_csv(dir_loc +  "gender_age_test.csv", dtype = {"device_id": np.str, "age": np.int8})
        self.test = pd.DataFrame(test['device_id'])
        print("Done with datasets...")
        # Mandatory cleaning and creating of lables
        #
        self.label = train['group']
        self.test_device_id = pd.DataFrame(test['device_id'])        
        return train, test
        
    def __read_data_sppt(self):
        events = pd.read_csv(dir_loc + "events.csv", dtype = {"device_id": np.str})
        app_events = pd.read_csv(dir_loc + "app_events.csv")
        app_labels = pd.read_csv(dir_loc+ "app_labels.csv")
        label_categories = pd.read_csv(dir_loc + "label_categories_v2.csv")
        brands = pd.read_csv(dir_loc + "phone_brand_device_model.csv", dtype = {"device_id": np.str}, encoding = "UTF-8")
        return events, app_events, app_labels, label_categories, brands

        
    def brand_device(self):
        '''
        merge the brand and device to get the classification results
        https://www.kaggle.com/dvasyukova/talkingdata-mobile-user-demographics/brand-and-model-based-benchmarks/notebook
        '''
        train, test = self.__read_data_tt()
        events, app_events, app_labels, label_categories, brands = self.__read_data_sppt()
        print("Before Merge - shape of train {} and test {}".format(train.shape, test.shape))
        train = train.merge(brands[['device_id','phone_brand','device_model']], how='left',on='device_id')
        train.drop_duplicates(subset='device_id', keep='first', inplace=True)

        test = test.merge(brands[['device_id','phone_brand','device_model']], how='left',on='device_id')
        test.drop_duplicates(subset='device_id', keep='first', inplace=True)
        # Mandatory cleaning
        #
        train = train.drop(['age','gender','group'], axis=1)

#        append to train and test
        self.train = self.train.merge(train, how='left', on='device_id')
        self.test = self.test.merge(test, how='left', on='device_id')

        print("After merge - shape of train {}, test {}, and label {}".format(self.train.shape, self.test.shape, len(self.label)))

    def event_count(self):
        '''
        count of the event for each device id
        '''
        train = self.train
        test = self.test
        events, app_events, app_labels, label_categories, brands = self.__read_data_sppt()
        print("Before Merge - shape of train {} and test {}".format(train.shape, test.shape))
        event_count = pd.DataFrame({'event_count' : events.groupby(["device_id"])['event_id'].count()}).reset_index()
        train = train.merge(event_count, how='left', on='device_id')
        train['event_count'].fillna(0.0, inplace=True)
        test = test.merge(event_count, how='left', on='device_id')
        test['event_count'].fillna(0.0, inplace=True)

#        append to train and test
        self.train = self.train.merge(train, how='left', on='device_id')
        self.test = self.test.merge(test, how='left', on='device_id')
        
        print("After merge - shape of train {}, test {}, and label {}".format(self.train.shape, self.test.shape, len(self.label)))

# Create the feature pipeline along with transformer_weights
class brand_model(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        lebrand = LabelEncoder().fit(X.phone_brand)
        X['brand'] = lebrand.transform(X.phone_brand)
#        combination of phone brand and phone model
        m = X.phone_brand.str.cat(X.device_model)
        lemodel = LabelEncoder().fit(m)
        X['model'] = lemodel.transform(m)
        X = X.drop(['phone_brand','device_model'], axis=1)
#        print X.head()
        return X

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
            print("Column(s) to select are {} :".format(self.col_names))
#            print(data_frame)
            return data_frame[self.col_names]

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
#        return X


def build_pipeline():
    pipe = Pipeline([("FU", FeatureUnion(
                    transformer_list=[
                                ("PL_brand", Pipeline([
                                            ('print_data1', PrintTransfrmer()),
                                            ("CS_brand", ColumnSelector(['phone_brand','device_model'])),
                                            ("brand_model", brand_model()),
                                            ("brand_clf", GradientBoostingClassifier()),
                                            ('print_data2', PrintTransfrmer()),
                                            ])),
                                ("PL_events", Pipeline([
                                            ('print_data1', PrintTransfrmer()),
                                            ("CS_event", ColumnSelector(['event_count'])),
                                            ("event_clf", RandomForestClassifier()),
                                            ('print_data2', PrintTransfrmer()),
                                            ]))
                                    ],
                    transformer_weights = {
                                "PL_brand": 1.0,
                                "PL_events": 1.0
                                })),
                    ("CLF", LogisticRegression())
                    ])
    pipe_params = {'FU__PL_brand__brand_clf__learning_rate':[0.01],
                   'FU__PL_brand__brand_clf__n_estimators':[100],
                    'FU__PL_brand__brand_clf__min_samples_leaf': [64]
                    }
    return pipe, pipe_params
    
if __name__=='__main__':
    data = dataset()
    data.brand_device()
    data.event_count()
    print('Performing Pipeline...')
    pipe, pipe_params = build_pipeline()
#    print('Fit Pipeline...')
#    pipe.fit(data_brand.train, data_brand.label)
    print(data.train.shape, data.test.shape, data.label.shape, data.test_device_id.shape)
    print("Perform GridSearch")
    grid_search = GridSearchCV(pipe, pipe_params, cv=2, scoring='log_loss', verbose=3)
    grid_search.fit(data.train, data.label)
    print("The best CV model parameters are: {}".format(grid_search.best_estimator_))    
    print("The best CV model score is: {}".format(grid_search.best_score_))
    print("Performing the predictions...")
#    y_hat = pipe.predict_proba(data_brand.test)
    y_hat = grid_search.predict_proba(data.test)
#    print(type(y_hat))
    #print(y_hat)
    create_submission(data.test_device_id,y_hat)









