import urllib2 
import pandas as pd 
from collections import OrderedDict 

class tableFramer:
    def __init__(self, url, dataFormat = 'dataframe'):
        self.url = url
        self.dataFormat = dataFormat
        
        opener = urllib2.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        self.response = opener.open(self.url)
    
    def __call__(self):

        table_data = [[cell.text for cell in row("td")] for row in BeautifulSoup(self.response.read())("tr")]
        
        if self.dataFormat.lower() == 'json':
            return json.dumps(OrderedDict(table_data))
        else:
            return pd.DataFrame(table_data)
copeData('https://web.mo.gov/doc/offSearchWeb/searchOffender.do?docId=512073','json')
print copeData()
