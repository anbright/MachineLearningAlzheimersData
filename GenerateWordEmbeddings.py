# Converts News Articles to word embeddings
import csv
import sys
csv.field_size_limit(sys.maxsize)

## Constants
vocab_size = 50000

## Loading Articles
## source: https://www.kaggle.com/snapcrack/all-the-news
print('Loading articles')

article_titles = []
article_contents = []

for s in ['articles1.csv', 'articles2.csv', 'articles3.csv']:
	with open(s) as f:
		reader = csv.reader(f)
		headers = reader.next()

		for row in reader:
			## Do not include Breitbart articles
			if row[3] != 'Breitbart':
				article_titles.append(row[2])
				article_contents.append(row[9])

print(len(article_titles), len(article_contents))

## Cleaning Data
print('Getting Glove Embeddings')

if not os.path.exists(glove_name):
    path = 'glove.6B.zip'
    path = get_file(path, origin="http://nlp.stanford.edu/data/glove.6B.zip")
    

# Lowercase

