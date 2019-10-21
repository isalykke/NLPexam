#code from: https://stackoverflow.com/questions/19056031/download-files-using-requests-and-beautifulsoup
import requests
from bs4 import BeautifulSoup as bs
import urllib
import re

def get_valid_filename(s):
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)

_URL = "https://www.grc.com/securitynow.htm"

# functional
r = requests.get(_URL)
soup = bs(r.text, features="lxml")
urls = []
names = []
for i, link in enumerate(soup.findAll('a')):
    _FULLURL = _URL + str(link.get('href'))
    if _FULLURL.endswith('.txt'):
        urls.append(_FULLURL.replace("securitynow.htm/", "")) #remove "securitynow.htm/" to get to the right page for txt files
        names.append(soup.select('a')[i].attrs['href'])

names_urls = zip(names, urls)

for name, url in names_urls:
    print(name, url)
    res = urllib.request.urlopen(url)
    txt = open(get_valid_filename(name), 'wb')
    txt.write(res.read())
    txt.close()

