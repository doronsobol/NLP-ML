#import wikipedia
import random
import re
from collections import defaultdict
import ipdb
import requests
from urlparse import urlparse
import goose

import sys
reload(sys)  # setdefaultencoding is purposely deleted after import so it must be reloaded
sys.setdefaultencoding('utf-8')

urls = [
#'nytimes',
'bbc.co.uk',
#'guardian'
]

ignore = [
'/programmes/', # https://en.wikipedia.org/wiki/Hercule_Poirot#cite_note-61
]
USER_AGENT = 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36'

article_cnt = 0
found_cnt = 0
id = 13000 # started with 1000 before
g = goose.Goose()

while found_cnt < 500:
    id += random.randint(1,20)  
    try:
        url = 'http://en.wikipedia.org/?curid='+str(id)
        print 'Requesting', url,
        req = requests.get(url, headers={'User-Agent' : USER_AGENT})
        if req.status_code != 200:
            print req.status_code
            continue
        response = req.text
        print 'Got response'
        '''
        except BaseException as e:
            #print url
            print e
            continue
        
        try:
        '''
        sents = re.findall('\. ([A-Z][^<.]*\.)<sup id="(.*?)".*?class="reference"', response)
        for sent, citation_id in sents:
            ref = re.search('href="#%s".*citation.*http://(.*?)">'% citation_id, response)
            if not ref: # TODO verify that this only happens for text sources and non-url stuff
                continue
            #print response
            #print 'href="#%s".*citation.*http://(.*?)">'% citation_id
            ref = ref.groups()[0]
            ref = ref.lower()
            if any(url in ref for url in urls) and not any(ign in ref for ign in ignore):
                with open(str(id)+'__'+ref[:25].replace('/', '')+'__'+sent[:15].replace(' ', '_')+'.txt', 'w') as file:
                    source_text = g.extract(url='http://'+ref).cleaned_text.replace('\n', ' ')
                    if len(source_text) < 10: # Shouldn't happen really...
                        print 'Skipped one because too short'
                        print ref
                    lines = [sent, ref, source_text]
                    file.write('\n'.join(lines))
                    found_cnt +=1

    except BaseException as e:
        #print url
        print e
        continue
                
    article_cnt +=1
    
    if article_cnt%100 == 0:
        print article_cnt, found_cnt



    
    