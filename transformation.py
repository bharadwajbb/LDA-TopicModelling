import sys, re
from gensim import corpora
from database import MongoDatabase
from helper_functions import *


class Transformation():

	def __init__(self):
		self.db = MongoDatabase()

	def transform_for_lda(self, save_folder = os.path.join('files', 'lda'), no_below = 5, no_above = 0.90):
		D = self.db.read_collection(collection = 'publications_raw')
		texts = [x['tokens'] for x in D]
		dictionary = corpora.Dictionary(texts)
		dictionary.filter_extremes(no_below = no_below, no_above = no_above)
		create_directory(save_folder)
		dictionary.save(os.path.join(save_folder, 'dictionary.dict'))
		corpus = [dictionary.doc2bow(text) for text in texts]
		corpora.MmCorpus.serialize(os.path.join(save_folder, 'corpus.mm'), corpus)
