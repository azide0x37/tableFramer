import requests
import json 
from bs4 import BeautifulSoup
from collections import OrderedDict 

class tableFramer:
    def __init__(self, url):
        self.url = url
        self.response = requests.get(url, headers = {'User-Agent': 'Mozilla/5.0'})

    def __call__(self):
        souped = BeautifulSoup(self.response.text)
        
        tableHead = souped.find('thead')
        colNames = tableHead.findAll('th')
        print "colNames", colNames

        table = souped.find('table', summary = "Table listing details of the accident.")
        rows = table.findAll('tr', class_ = "infoCell")
        print "rows", rows
        
        dataset = []

        for tr in rows:
            cols = tr.findAll('td')
            rowData = OrderedDict()
            counter = 1

            for td in cols[1:]:
                text = ''.join(td.find(text=True))
                try:
                    rowData[colNames[counter]] = text
                    counter += 1
                except:
                    counter = 0
                    continue
            dataset.append(rowData)

        return json.dumps(dataset)#, indent=4, separators=(',',':'))
        
crashData = tableFramer('http://www.mshp.dps.missouri.gov/HP68/SearchAction')
print crashData()
