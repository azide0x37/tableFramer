import urllib2 
import json
import pandas as pd 
from bs4 import BeautifulSoup
from collections import OrderedDict 

class tableFramer:
    def __init__(self, url, dataFormat = 'dataframe'):
        self.url = url
        self.dataFormat = dataFormat
        
        opener = urllib2.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        self.response = opener.open(self.url)
    
    def __call__(self):
        souped = BeautifulSoup(self.response.read())
        tables = souped.findAll('table')
        table_data = [[cell.text for cell in row("td")] for row in tables]
        
        if self.dataFormat.lower() == 'json':
            return json.dumps(OrderedDict(table_data))
        else:
            return pd.DataFrame(table_data)

copeData = tableFramer('https://web.mo.gov/doc/offSearchWeb/searchOffender.do?docId=512073')
print copeData()
