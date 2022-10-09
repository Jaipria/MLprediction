import os

urls = open('https://lua.rlp.de/de/unsere-themen/infektionsschutz/meldedaten-coronavirus/meldedaten-excel/', 'r')
for i, url in enumerate(urls):
    path = 'E:\Python\Research Lab\{}'.format(os.path.basename(url)
    urllib.request.urlretrieve(url, path)
    