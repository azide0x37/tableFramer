import urllib2
import requests
import json
import pandas as pd 
from bs4 import BeautifulSoup
from collections import OrderedDict 

class tableFramer:
    def __init__(self, url, dataFormat = 'dataframe'):
        self.url = url
        self.dataFormat = dataFormat
        self.headers = {'User-Agent': 'Mozilla/5.0'}
        
        #request conversion template
        self.response = requests.get(url, headers=headers)
        
        #legacy urllib2
        """
        opener = urllib2.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        self.response = opener.open(self.url)
        """
    
    def __call__(self):
        souped = BeautifulSoup(self.response.text)
        tables = souped.findAll('table')
        table_data = [[cell.text for cell in row("td")] for row in tables]
        
        if self.dataFormat.lower() == 'json':
            return json.dumps(OrderedDict(table_data))
        else:
            return pd.DataFrame(table_data)

copeData = tableFramer('https://web.mo.gov/doc/offSearchWeb/searchOffender.do?docId=512073')
print copeData()
