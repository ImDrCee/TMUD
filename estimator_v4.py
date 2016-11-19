"""
Created on Sun Jul 24 20:38:01 2016

@author: nchandra
"""

"""
Will not compbine all columns in one feature.
Combine feature one at a time.
Build and test classifiers for each feature combination.
Finally combine all classifiers in a pipeline
Read each file into the object and then create features as required.
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
#from sklearn.linear_model import LogisticRegression
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
    def __init__(self):
        self.train = []
        self.label = []
        self.test = []
        self.test_device_id = []
        
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
        
    def brand_device_merge(self):
        '''
        merge the brand and device to get the classification results
        https://www.kaggle.com/dvasyukova/talkingdata-mobile-user-demographics/brand-and-model-based-benchmarks/notebook
        '''
        train, test, events, app_events, app_labels, label_categories, brands = self.__read_data()
        print("Before Merge - shape of train {} and test {}".format(train.shape, test.shape))
        self.train = train.merge(brands[['device_id','phone_brand','device_model']], how='left',on='device_id')
        self.train.drop_duplicates(subset='device_id', keep='first', inplace=True)
        self.test = test.merge(brands[['device_id','phone_brand','device_model']], how='left',on='device_id')
        self.test.drop_duplicates(subset='device_id', keep='first', inplace=True)
        # Mandatory cleaning and creating of lables
        #
        self.label = self.train['group']
        self.train = self.train.drop(['device_id','age','gender','group'], axis=1)
        self.test_device_id = test['device_id']
        self.test = self.test.drop(['device_id'], axis=1)
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

def build_pipeline():
    pipe = Pipeline([("brand_model", brand_model()),
    ("clf", GradientBoostingClassifier())
    ])
    pipe_params = {'clf__learning_rate':[0.0001, 0.001],
                   'clf__n_estimators':[100],
                    'clf__min_samples_leaf': [64, 96, 128]

    }
    return pipe, pipe_params

if __name__=='__main__':
    data_brand = dataset()
    data_brand.brand_device_merge()
    print('Performing Pipeline...')
    pipe, pipe_params = build_pipeline()
#    print('Fit Pipeline...')
#    pipe.fit(data_brand.train, data_brand.label)
    print("Perform GridSearch")
    grid_search = GridSearchCV(pipe, pipe_params, cv=2, scoring='log_loss', verbose=3)
    grid_search.fit(data_brand.train, data_brand.label)
    print("The best CV model parameters are: {}".format(grid_search.best_estimator_))    
    print("The best CV model score is: {}".format(grid_search.best_score_))
    print("Performing the predictions...")
#    y_hat = pipe.predict_proba(data_brand.test)
    y_hat = grid_search.predict_proba(data_brand.test)
#    print(type(y_hat))
    #print(y_hat)
    create_submission(data_brand.test_device_id,y_hat)









