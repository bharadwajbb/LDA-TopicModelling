import sys, re
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from database import MongoDatabase
from helper_functions import *



class Evaluation():

	def __init__(self):
		self.db = MongoDatabase()

	def calculate_coherence(self, file_folder = os.path.join('files', 'lda'), models_folder = os.path.join('files', 'models')):
		dictionary, corpus = get_dic_corpus(file_folder)
		texts = [x['tokens'] for x in self.db.read_collection('publications_raw')]
		M = [x for x in read_directory(models_folder) if x.endswith('lda.model')]
		processed_models = ['{}-{}-{}-{}-{}'.format(x['k'], x['dir_prior'], x['random_state'], x['num_pass'], x['iteration']) for x in self.db.read_collection('coherence')]
		for i, m in enumerate(M):
			k = m.split(os.sep)[2]
			dir_prior = m.split(os.sep)[3]
			random_state = m.split(os.sep)[4]
			num_pass = m.split(os.sep)[5]
			iteration = m.split(os.sep)[6]

			if '{}-{}-{}-{}-{}'.format(k, dir_prior, random_state, num_pass, iteration) not in processed_models: 
				model = models.LdaModel.load(m)
				coherence_c_v = CoherenceModel(model = model, texts = texts, dictionary = dictionary, coherence='c_v')
				score = coherence_c_v.get_coherence()
				doc = {	'k' : k, 'dir_prior' : dir_prior, 'random_state' : random_state, 'num_pass' : num_pass, 'iteration' : iteration, 'coherence_score' : score}
				self.db.insert_one_to_collection('coherence', doc)
			else:
				continue


	def plot_coherence(self, min_k = 2, max_k = 20, save_location = os.path.join('files', 'plots'), plot_save_name = 'coherence_scores_heatmap.pdf'):
		create_directory(save_location)
		D = list(self.db.read_collection(collection = 'coherence'))
		data = [[int(x['k']), x['dir_prior'],x['random_state'], x['num_pass'], x['iteration'], x['coherence_score']] for x in D]
		df = pd.DataFrame()
		for k in range(min_k, max_k + 1):
			df_temp = pd.DataFrame(index = [k])
			for row in sorted(data):
				if row[0] == k:
					df_temp['{}-{}-{}-{}'.format(row[1],row[2],row[3],row[4])] = pd.Series(row[5], index=[k])
			
			df = df.append(df_temp)
		
		df = df.transpose()
		ax = sns.heatmap(df, cmap = "Blues", annot = True, vmin = 0.500, vmax = 0.530, square = True, annot_kws = {"size": 11},
							fmt = '.3f', linewidths = .5, cbar_kws = {'label': 'coherence score'})

		ax.xaxis.tick_top()
		plt.yticks(rotation=0)
		plt.xticks(rotation=0, ha = 'left') 
		fig = ax.get_figure()
		fig.set_size_inches(19, 6)
		fig.savefig(os.path.join(save_location, plot_save_name), bbox_inches='tight')



	def output_lda_topics(self, K = 9, dir_prior = 'auto', random_state = 42, num_pass = 15, iteration = 200, top_n_words = 10, models_folder = os.path.join('files', 'models'), 
						save_folder = os.path.join('files', 'tables')):

		model = load_lda_model(os.path.join(models_folder, str(K), dir_prior, str(random_state), str(num_pass), str(iteration)))
		topic_table, topic_list = [], []
		for k in range(K):
			topic_table.append(['{}'.format(get_topic_label(k, labels_available = False).upper())])
			topic_table.append(["word", "prob."])
			list_string = ""
			topic_string = ""
			topic_string_list = []

			scores = model.print_topic(k, top_n_words).split("+")
			for score in scores:
				score = score.strip()
				split_scores = score.split('*')
				percentage = split_scores[0]
				word = split_scores[1].strip('"')
				topic_table.append([word.upper(), "" + percentage.replace("0.", ".")])
				list_string += word + ", "

			topic_table.append([""])
			topic_list.append([str(k+1), list_string.rstrip(", ")])

		save_csv(topic_list, 'topic-list', folder = save_folder)
		save_csv(topic_table, 'topic-table', folder = save_folder)

