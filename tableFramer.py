import requests
import json
import pandas as pd 
from bs4 import BeautifulSoup
from collections import OrderedDict 

class tableFramer:
    def __init__(self, url, dataFormat = 'dataframe'):
        self.url = url
        self.dataFormat = dataFormat.lower()
        self.headers = {'User-Agent': 'Mozilla/5.0'}
        self.response = requests.get(url, headers=self.headers)

    def __call__(self):
        souped = BeautifulSoup(self.response.text)
        tables = souped.findAll('table')
        table_data = [[cell.text for cell in row("td")] for row in tables]
        
        if self.dataFormat == 'json':
            return json.dumps(table_data)
        else:
            return pd.DataFrame(table_data)

copeData = tableFramer('https://web.mo.gov/doc/offSearchWeb/searchOffender.do?docId=512073')
print copeData()
