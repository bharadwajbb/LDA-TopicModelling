import os, requests, textract, glob2, sys, csv
from datetime import datetime
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim import corpora, models


def create_directory(name):
	try:
		if not os.path.exists(name):
			os.makedirs(name)
	except (Exception):
		exit(1)


def get_HTTPHeaders():
	return {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit 537.36 (KHTML, like Gecko) Chrome", 
				"Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
				"Accept-Language": "en-US,en;q=0.5"}


def return_html(url):
	try:
		html = requests.get(url, headers=get_HTTPHeaders())
		if html.status_code == requests.codes.ok:
			return html
		else:
			return None
	except (Exception):
		return None


def save_pdf(url, folder, name, overwrite = True):
	create_directory(folder)
	file_exists = os.path.exists(os.path.join(folder, name))
	if overwrite == True or file_exists == False:
		try:
			response = requests.get(url, headers= get_HTTPHeaders(), stream=True)
			with open('{}/{}'.format(folder, name), 'wb') as f:
				f.write(response.content)
		except (Exception):
			exit(1)

def pdf_to_plain(pdf_file):
	try:
		return textract.process(pdf_file)
	except (Exception):
		return None
		

def read_directory(directory):
	try:
		return glob2.glob(os.path.join(directory, '**' , '*.*'))
	except (Exception):
		exit(1)



def word_tokenizer(text):
	try:
		return  [token.lemma_ for token in text if token.is_alpha and not token.is_stop and len(token) > 1]
	except (Exception):
		exit(1)


def get_bigrams(text):
	try:
		return list(nltk.bigrams(text.split()))
	except (Exception):
		exit(1)


def named_entity_recognition(text):
	try:
		ents = text.ents
		entities = [str(entity).lower() for entity in ents if len(str(entity).split()) > 2]
		return [ent.strip() for ent in entities if not any(char.isdigit() for char in ent) and all(ord(char) < 128 for char in ent)]
	except (Exception):
		exit(1)


def get_dic_corpus(file_folder):
	dic_path = os.path.join(file_folder, 'dictionary.dict')
	corpus_path = os.path.join(file_folder, 'corpus.mm')
	if os.path.exists(dic_path):
		dictionary = corpora.Dictionary.load(dic_path)
	else:
		exit(1)
	if os.path.exists(corpus_path):
		corpus = corpora.MmCorpus(corpus_path)
	else:
		exit(1)
	return dictionary, corpus


def load_lda_model(model_location):
	model_path = os.path.join(model_location, 'lda.model')
	if os.path.exists(model_path):
		return  models.LdaModel.load(model_path)
	else:
		exit(1)


def get_topic_label(k, labels_available = True):
	if not labels_available:
		return 'Topic {}'.format(k)
	else:
		topics = {	0 : 'Convergence',
					1 : 'State, Policy, Action',
					2 : 'Linear Algebra',
					3 : 'NLP',
					4 : 'Inference',
					5 : 'Computer Vision',
					6 : 'Graphical Models',
					7 : 'Neural Network Learning',
					8 : 'Stimulus Response',
					9 : 'Neural Network Structure'}
		
		return topics[k]


def save_csv(data, name, folder):
	try:
		create_directory(folder)
		suffix = '.csv'
		if name[-4:] != suffix:
			name += suffix
		path = os.path.join(folder, name)
		with open(path, "w") as f:
			writer = csv.writer(f, lineterminator='\n')
			writer.writerows(data)
	except (Exception):
		exit(1)


def read_csv(filename, folder = None):
	if folder is not None:
		filename = os.path.join(folder, filename)
	try:
		csv.field_size_limit(sys.maxsize)
		with open(filename, 'rb') as f:
			reader = csv.reader(f)
			return list(reader)
	except (Exception):
		exit(1)