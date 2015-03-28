import urllib2 
import pandas as pd 
from collections import OrderedDict 
from datetime import datetime 

class tableFramer:
    def __init__(self, url, dataFormat = dataframe):
        self.url = url
        self.dataFormat = dataFormat
    
    def __call__(self):

        allTableData = [{y.findAll('tr'):x} for y in x.findAll('td') for x in webpageSouped.findAll('table')]
    
        heading = [x.findAll('tr') for x in webpageSouped.findAll('table')]
        data = [y.findall('td') for y in heading]
    
        dataset = {x:y for x in heading for y in data}
        
        for tr in rows:
            cols = OrderedDict()
            counter = 0
    .  
        for td in cols[1:]:
            text = ''.join(td.find(text=True))
            try:
                headings[counter] = text
                counter += 1
            except:
                counter = 0
                continue
            
            dataset.append(row_data)
      
        return pd.DataFrame(dataset)
