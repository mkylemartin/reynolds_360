from bs4 import BeautifulSoup
import time
import requests as r
import re
from glob import glob

todo = ['aa.html', ]
regex = r'(..\.html)'
already_scraped = []


def get_rnlp(address):
    '''
    pass in 'aa.html' like string returns a receipt
    writes a file to a directory.
    '''
    rob = 'http://reynoldsnlp.com/scrape/'
    webpage = rob + address
    headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_2) '
                             'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63'
                             '.0.3239.132 Safari/537.36'}
    response = r.get(webpage, headers=headers)
    with open('scrape/' + address, 'w+') as f:
        f.write(response.text)
        time.sleep(1)
    return print('written: ' + address)


def get_hrefs(address):
    '''
    pass in 'aa.html' and this will go get the hrefs in the given html file
    searching in the /scrape directory
    '''
    hrefs = []
    with open('scrape/' + address, 'r') as f:
        soup = BeautifulSoup(f, 'html5lib')
        for link in soup.find_all('a'):
            hrefs.append(re.findall(regex, link.get('href'))[0])
    return hrefs

# get html from item in todo
get_rnlp(todo[0])
# get the links from new item
new = get_hrefs(todo[0])
# add new items from todo[0] to todo
for item in new:
    todo.append(item)
# add page to already scraped and delete it from todo
already_scraped.append(todo[0])
todo.remove(todo[0])

print(todo)
print(already_scraped)

while True:
    if todo == []:
        break
    else:
        for link in todo:
            if link in already_scraped:
                todo.remove(link)
            else:
                get_rnlp(link)
                new_hrefs = get_hrefs(link)

                for href in new_hrefs:
                    if href in todo or href in already_scraped:
                        continue
                    else:
                        todo.append(href)
                already_scraped.append(link)
                print('len(todo): ' + str(len(todo)))
                print('len(already_scraped): ' + str(len(already_scraped)))


print('yay, all finished!')
