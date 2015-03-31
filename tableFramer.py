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
        
        headerTable = souped.find('thead')
        colNames = headerTable.findAll('th')
        table = souped.find('table', summary="Table listing details of the accident.")
        rows = table.findAll('tr')
        dataset = []
        
        for tr in rows:
            cols = tr.findAll('td')
            row_data = OrderedDict()
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

        if self.dataFormat != 'dataframe':
            return json.dumps(dataset, indent=4, separators=(',',':'))
        else:
            return pd.DataFrame(table_data)

copeData = tableFramer('http://www.mshp.dps.missouri.gov/HP68/SearchAction','json')
print copeData()
