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
        table = souped.find('table', summary="Table listing details of the accident.")
        #table_data = [[cell.text for cell in row("td")] for row in table]
        column_names = [x for x in xrange(0, len(table.findAll('tr')))]
        dataset = []
        columns = [[{column_names[int(text)]:text.find(text=True)} for text in row.findAll('td')] for row in table.findAll('tr')]
        
        
            
        
        
        
        
        if self.dataFormat != 'dataframe':
            return json.dumps(table_data[1])
        else:
            return pd.DataFrame(table_data)

copeData = tableFramer('http://www.mshp.dps.missouri.gov/HP68/SearchAction','json')
print copeData()
