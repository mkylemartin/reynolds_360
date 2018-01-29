import re
webpage = open('scraped_wikipedia_article.html', 'r')
txt_webpage = webpage.read()

my_regex = r'<p>(.*?)</p>'

p_tags = re.findall(my_regex, txt_webpage)
# print(len(p_tags))

with open('scraped_wikipedia_article_parsed.txt', 'w+') as parsed_file:
	n = 1
	for item in p_tags:
		parsed_file.write('Item number ' + str(n) + ':\n\n')
		parsed_file.write(item)
		parsed_file.write('\n\n')
		n += 1
