import re
webpage = open('scraped_wikipedia_article.html', 'r')
txt_webpage = webpage.read()

my_regex = r'<p>(.*)</p>'

p_tags = re.findall(my_regex, txt_webpage)
print(len(p_tags))

f = open('scraped_wikipedia_article_parsed.txt', 'w+')

n = 1
for item in p_tags:
	f.write('Item number ' + str(n) + ':\n\n')
	f.write(item)
	f.write('\n\n')
	n += 1
f.close()
