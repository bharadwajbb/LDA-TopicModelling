

# packages and modules
import sys, spacy
from collections import Counter
import itertools
from database import MongoDatabase
from helper_functions import *


class Preprocessing():

	def __init__(self):
		self.db = MongoDatabase()

	def full_text_preprocessing(self, pdf_folder = os.path.join('files', 'pdf')):
		F = [x for x in read_directory(pdf_folder) if x[-4:] == '.pdf']
		processed_documents = [ '{}-{}-{}'.format(x['journal'], x['year'], x['title']) for x in self.db.read_collection('publications_raw')]
		print(processed_documents)
		for i, f in enumerate(F):
			journal = f.split('\\')[2]
			year = f.split('\\')[3]
			title = f.split('\\')[4]

			if '{}-{}-{}'.format(journal, year, title) in processed_documents:
				continue

			content = pdf_to_plain(f)
			if content is not None:
				content = content.decode('utf-8')
				content = content.replace(u'\xad', "-")
				content = content.replace(u'\u2014', "-")
				content = content.replace(u'\u2013', "-")
				content = content.replace(u'\u2212', "-")
				content = content.replace('-\n','')
				content = content.replace('\n',' ')
				content = content.replace(u'\ufb02', "fl")
				content = content.replace(u'\ufb01', "fi")
				content = content.replace(u'\ufb00', "ff")
				content = content.replace(u'\ufb03', "ffi")
				content = content.replace(u'\ufb04', "ffl")
				if content.rfind("References") > 0:
					content = content[:content.rfind("References")]
				if content.rfind("Acknowledgment") > 0:
					content = content[:content.rfind("Acknowledgment")]
				doc = {	'journal' : journal, 'title' : title, 'year' : year, 'content' : content}
				print(doc)
				self.db.insert_one_to_collection(doc = doc, collection = 'publications_raw')

	def general_preprocessing(self, min_bigram_count = 5):
		D = self.db.read_collection(collection = 'publications_raw')
		nlp = setup_spacy()
		for i, d in enumerate(D):
			if d.get('tokens') is None:
				content = nlp(d['content'])
				unigrams = word_tokenizer(content)
				entities = named_entity_recognition(content)
				bigrams = get_bigrams(" ".join(unigrams))
				bigrams = [['{} {}'.format(x[0],x[1])] * y for x, y in Counter(bigrams).most_common() if y >= min_bigram_count]
				bigrams = list(itertools.chain(*bigrams))
				d['tokens'] = unigrams + bigrams + entities

				self.db.update_collection(collection = 'publications_raw', doc = d)
			else:
				print('Document already tokenized, skipping ...')

def setup_spacy():
	nlp = spacy.load('en_core_web_sm')
	for word in set(stopwords.words('english')):
		nlp.Defaults.stop_words.add(str(word))
		nlp.Defaults.stop_words.add(str(word.title()))
	for word in nlp.Defaults.stop_words:
		lex = nlp.vocab[word]
		lex.is_stop = True
	return nlp

			