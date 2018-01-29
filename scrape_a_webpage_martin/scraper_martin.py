import requests as r

url = 'https://en.wikipedia.org/wiki/Famous_for_being_famous'
headers = {'user-agent': 'mkmartin (kyle.martin@byu.edu)'}
response = r.get(url, headers=headers)

print(response.text[:100], '....')
print()

f = open("scraped_wikipedia_article.html", "w+")
f.write(response.text)
f.close()
