import urllib2 
import pandas as pd 
from collections import OrderedDict 
from datetime import datetime 

class tableFramer:
  def __init__(self, url, format = dataframe):
    self.url = url
    self.format = format
    
  def __call__(self):
    headingTable = webpageSouped.find('table',,)
    headings = headingTable.findAll('td')
    
    dataTable = webpageSouped.find('table', border="0", cellpadding="4", cellspacing="2")
    rows = dataTable.findAll('tr')
    
    dataset = []
    
    for tr in rows:
      cols = OrderedDict()
      counter = 0
      
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
