import urllib2 
import pandas as pd 
from collections import OrderedDict 
from datetime import datetime 

class tableFramer:
  def __init__(self, url, dataFormat = dataframe):
    self.url = url
    self.dataFormat = dataFormat
    
  def __call__(self):
    allTable = webpageSouped.findAll('table')
    [x.findAll('td') for x in headingTable]
    allTableData = [y.findAll('tr') for y in [x.findAll('td') for x in headingTable]]
    
    dataTable = webpageSouped.find('table', border="0", cellpadding="4", cellspacing="2")
    rows = dataTable.findAll('tr')
    
    dataset = []
    
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
