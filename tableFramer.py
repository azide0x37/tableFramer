tableFramer.py
#web analysis
import requests
import urllib2
import json 
from bs4 import BeautifulSoup
from collections import OrderedDict 

#machine learning
import numpy as np
import pandas as pd
from sklearn import pipeline
from sklearn import datasets
from sklearn import metrics
from sklearn.qda import QDA
import sklearn.preprocessing
import sklearn.cross_validation
import sklearn.decomposition

#sklearn_pandas, goddamnit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import cross_validation
from sklearn import grid_search
import sys

"""
dataset = datasets.load_iris()

model = QDA()
model.fit(dataset.data, dataset.target)
print (model)
 
expected = dataset.target
predicted = model.predict(dataset.data)
 
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
"""

class mshpScraper:
    
    def __init__(
                self,
                url = 'http://www.mshp.dps.missouri.gov/HP68/SearchAction', 
                colNames = ['Name', 'Age', 'Hometown', 'Severity', 'Date', 'Time', 'County', 'Location', 'Troop'], 
                county = 'Jefferson'):

        self.url = url
        self.colNames = colNames
        
        #Use a proper useragent to evade the anti-hack software
        opener = urllib2.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        
        #Cache our response
        self.response = opener.open(self.url)
    
        #Set better formatting for testing console print
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 200)
    
    def __str__(self):
        try:
            return self.url
        except:
            print "Unable to print url." 
    
    #Setting this default method allows one to call an instance like a function. NEATO!
    def __call__(self, refreshCache = False):
        
        if refreshCache:
            self.__init__()
        
        webpageSouped = BeautifulSoup(self.response.read())

        #This is the particular table with the data we need.
        #TODO: Find a better way to pick the right one?
        table = webpageSouped.find('table', summary="Table listing details of the accident.")
        rows = table.findAll('tr')
        
        dataset = []
        
        for tr in rows[1:]:
            cols = tr.findAll('td')
            row_data = {}
            counter = 0

            for td in cols[1:]:
                text = ''.join(td.find(text=True))
                try:
                    row_data[self.colNames[counter]] = text
                    counter += 1
                except:
                    counter = 0
                    continue
            dataset.append(row_data)

        #Convert dataset (List of Dicts) to pandas DataFrame
        returnData = pd.DataFrame(dataset)
        returnData = returnData.drop(['Name', 'Hometown', 'Date', 'Time', 'Location'], axis=1)

        #Ship it!
        return returnData.transpose()

myScrape = mshpScraper()
data = myScrape()

if sys.version_info >= (3, 0):
    basestring = str

def cross_val_score(model, X, *args, **kwargs):
    X = DataWrapper(X)
    return cross_validation.cross_val_score(model, X, *args, **kwargs)


class GridSearchCV(grid_search.GridSearchCV):
    def fit(self, X, *params, **kwparams):
        super(GridSearchCV, self).fit(DataWrapper(X), *params, **kwparams)

    def predict(self, X, *params, **kwparams):
        super(GridSearchCV, self).fit(DataWrapper(X), *params, **kwparams)

try:
    class RandomizedSearchCV(grid_search.RandomizedSearchCV):
        def fit(self, X, *params, **kwparams):
            super(RandomizedSearchCV, self).fit(DataWrapper(X), *params, **kwparams)

        def predict(self, X, *params, **kwparams):
            super(RandomizedSearchCV, self).fit(DataWrapper(X), *params, **kwparams)
except AttributeError:
    pass


class DataWrapper(object):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, key):
        return self.df.iloc[key]


class PassthroughTransformer(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        return np.array(X).astype(np.float)


class DataFrameMapper(BaseEstimator, TransformerMixin):
    '''
    Map Pandas data frame column subsets to their own
    sklearn transformation.
    '''

    def __init__(self, features):
        '''
        Params:
        features    a list of pairs. The first element is the pandas column
                    selector. This can be a string (for one column) or a list
                    of strings. The second element is an object that supports
                    sklearn's transform interface.
        '''
        self.features = features


    def _get_col_subset(self, X, cols):
        '''
        Get a subset of columns from the given table X.
        X       a Pandas dataframe; the table to select columns from
        cols    a string or list of strings representing the columns
                to select
        Returns a numpy array with the data from the selected columns
        '''
        return_vector = False
        if isinstance(cols, basestring):
            return_vector = True
            cols = [cols]

        if isinstance(X, list):
            X = [x[cols] for x in X]
            X = pd.DataFrame(X)

        elif isinstance(X, DataWrapper):
            # if it's a datawrapper, unwrap it
            X = X.df

        if return_vector:
            t = X[cols[0]]
        else:
            t = X.as_matrix(cols)

        return t


    def fit(self, X, y=None):
        '''
        Fit a transformation from the pipeline
        X       the data to fit
        '''
        for columns, transformer in self.features:
            if transformer is not None:
                transformer.fit(self._get_col_subset(X, columns))
        return self


    def transform(self, X):
        '''
        Transform the given data. Assumes that fit has already been called.
        X       the data to transform
        '''
        extracted = []
        for columns, transformer in self.features:
            # columns could be a string or list of
            # strings; we don't care because pandas
            # will handle either.
            if transformer is not None:
                fea = transformer.transform(self._get_col_subset(X, columns))
            else:
                fea = self._get_col_subset(X, columns)

            if hasattr(fea, 'toarray'):
                # sparse arrays should be converted to regular arrays
                # for hstack.
                fea = fea.toarray()

            if len(fea.shape) == 1:
                fea = np.array([fea]).T
            extracted.append(fea)

        # combine the feature outputs into one array.
        # at this point we lose track of which features
        # were created from which input columns, so it's
        # assumed that that doesn't matter to the model.
        return np.hstack(extracted)

"""
names = datar[0]
print names
my_names = names.keys()
formats = [type(name) for name in names]
dtype = dict(names = names, formats = formats)
print dtype
array = np.array(names.items(), dtype=dtype)
print array

myScrape = mshpScraper()
data = myScrape()
"""
data = pd.DataFrame({'pet': ['cat', 'dog', 'dog', 'fish', 'cat', 'dog', 'cat', 'fish'], 
'children': [4., 6, 3, 3, 2, 3, 5, 4], 
'salary':   [90, 24, 44, 27, 32, 59, 36, 27]})

data2 = pd.DataFrame({
'Age': ['22','33','44','44','23','12','56','44','23','78','32','67','56','45','62','83','63','50','59','22','21',],
'County': ['Linn','Iron','Jefferson','Jefferson','Clay','Clay','Ray','Ray','Iron','Madison','Madison','Madison','Clay','Ray','Johnson','Johnson','Franklin','Jefferson','Jefferson','Jefferson','Jefferson',],
'Severity': ['Minor','Minor','Minor','Minor','Minor','Minor','Minor','Minor','Serious','Serious','Moderate','Moderate','Moderate','Major','Major','Major','Minor','Minor','Serious','Minor','Fatal',],
'Troop': ['C','C','B','A','A','I','J','H','H','J','D','D','B','I','J','F','F','F','G','E','E',]})


mapper = DataFrameMapper([
    ('pet', sklearn.preprocessing.LabelBinarizer()),
    ('children', sklearn.preprocessing.StandardScaler()),
    ('salary', sklearn.preprocessing.StandardScaler()),
    ])
    
mapper2 = DataFrameMapper([
    ('Age', sklearn.preprocessing.StandardScaler()),
    ('Severity', sklearn.preprocessing.LabelBinarizer()),
    ('County', sklearn.preprocessing.LabelBinarizer()),
    ('Troop', sklearn.preprocessing.LabelBinarizer())
    ])
    
#print data
#print mapper.features
print np.round(mapper2.fit_transform(data2), 2)
#pipe = sklearn.pipeline.Pipeline([('featurize', mapper2), ('lm', sklearn.linear_model.LinearRegression())])
#np.round(cross_val_score(pipe, data2), 2)
print mapper2
