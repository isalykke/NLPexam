#code from: https://stackoverflow.com/questions/19056031/download-files-using-requests-and-beautifulsoup
import requests
from bs4 import BeautifulSoup as bs
import re
import urllib
import os

def get_valid_filename(s):
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)


_URL = "https://www.grc.com/securitynow.htm" #current site
_URLARCHIVE = "https://www.grc.com/sn/past/year.htm" #generic site for older episodes
_ALLURLs = []

for i in range(2005, 2018):
    _ALLURLs.append(_URLARCHIVE.replace("year", str(i)))

_ALLURLs.append(_URL) #append newest episodes to list of older episodes

# functional
urls = []
names = []

for url in _ALLURLs:
    r = requests.get(url)
    soup = bs(r.text, features="lxml")
    for i, link in enumerate(soup.findAll('a')):
        _FULLURL = url + str(link.get('href'))
        if _FULLURL.endswith('.txt'):
            urls.append("https://www.grc.com" + soup.select('a')[i].attrs['href']) #append episidename to grc website
            names.append(soup.select('a')[i].attrs['href'])

names_urls = zip(names, urls)

os.chdir("/Users/isalykkehansen/OneDrive/uni_kandidat/natural_language_processing_NLP/NLPexam/txts") #change directory to txts folder for saving files

for name, url in names_urls:
    print(name, url)
    res = urllib.request.urlopen(url)
    txt = open(get_valid_filename(name), 'wb')
    txt.write(res.read())
    txt.close()

os.chdir("/Users/isalykkehansen/OneDrive/uni_kandidat/natural_language_processing_NLP/NLPexam/")


