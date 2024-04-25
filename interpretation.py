import sys, re, matplotlib
from operator import itemgetter
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
import pandas as pd
import numpy as np
from database import MongoDatabase
from helper_functions import *



class Interpretation():

	def __init__(self):
		self.db = MongoDatabase()
		self.plot_save_folder = os.path.join('files', 'plots')
		self.table_save_folder = os.path.join('files', 'tables')

	def infer_document_topic_distribution(self, K = 10, dir_prior = 'auto', random_state = 42, num_pass = 15, iteration = 200, top_n_words = 10, 
										models_folder = os.path.join('files', 'models'), lda_files_folder = os.path.join('files', 'lda')):

		dictionary, corpus = get_dic_corpus(lda_files_folder)
		model = load_lda_model(os.path.join(models_folder, str(K), dir_prior, str(random_state), str(num_pass), str(iteration)))
		D = self.db.read_collection(collection = 'publications_raw')
		for i, d in enumerate(D):
			if d.get('tokens') is not None:
				bow = model.id2word.doc2bow(d['tokens'])
				topics = model.get_document_topics(bow, per_word_topics = False)
				dic_topics = {}
				for t in topics:
					dic_topics[str(t[0])] = float(t[1])
				insert_doc = {'journal': d['journal'], 'year' : d['year'], 'title' : d['title'], 'topics' : dic_topics}
				self.db.insert_one_to_collection('publications', insert_doc)


	def get_document_title_per_topic(self):
		D = self.db.read_collection(collection = 'publications')
		titles = []

		for i, d in enumerate(D):
			dominant_topic = max(d['topics'].iteritems(), key = itemgetter(1))
			dominant_topic_id, dominant_topic_percentage = dominant_topic[0], dominant_topic[1]
			titles.append([d['year'], d['title'], d['journal'], dominant_topic_id, dominant_topic_percentage])
		save_csv(titles, 'titles-to-topics', folder = self.table_save_folder)


	def plot_topics_over_time(self, plot_save_name = 'topics-over-time.pdf'):
		D = self.db.read_collection(collection = 'publications')
		year_to_topics = get_year_to_topics(D)
		year_to_cum_topics = get_year_to_cum_topics(year_to_topics)
		df = pd.DataFrame.from_dict(year_to_cum_topics)
		fig, axs = plt.subplots(2,5, figsize=(15, 10))
		axs = axs.ravel()

		for index, row in df.iterrows():
			x = df.columns.values.tolist()
			y = row.tolist()

			axs[index].plot(x, y, 'o--', color='black', linewidth=1, label="Topic prevalence")
			axs[index].set_title(get_topic_label(index), fontsize=14)
			axs[index].set_ylim([0,0.4])

		plt.savefig(os.path.join(self.plot_save_folder, plot_save_name), bbox_inches='tight')
		plt.close()

	def plot_topics_over_time_stacked(self, plot_save_name = 'topics-over-time-stacked.pdf'):
		D = self.db.read_collection(collection = 'publications')
		year_to_topics = get_year_to_topics(D)
		year_to_cum_topics = get_year_to_cum_topics(year_to_topics)
		df = pd.DataFrame.from_dict(year_to_cum_topics)
		df = df.transpose()
		df.columns = [get_topic_label(x) for x in df.columns.values]
		ax = df.plot(figsize = (15, 8), kind = 'area', colormap='Spectral_r', rot = 45, grid = False)
		plt.xticks(df.index)
		plt.xlim(min(df.index), max(df.index))
		plt.ylim(0,1)
		handles, labels = ax.get_legend_handles_labels()
		plt.legend(reversed(handles), reversed(labels), loc = 'right', bbox_to_anchor=(1.35, 0.50), ncol=1, fancybox=False, shadow=False, fontsize=16)
		plt.savefig(os.path.join(self.plot_save_folder, plot_save_name), bbox_inches='tight')
		plt.close()


	def plot_topic_co_occurrence(self, plot_save_name = 'topic-co-occurrence.pdf'):
		D = self.db.read_collection(collection = 'publications')
		dominant_id_to_topics = {}

		for d in D:
			topics = [value for key, value in sorted(d['topics'].iteritems(), key=lambda x: int(x[0]))]
			max_topic_id = topics.index(max(topics))
			if max_topic_id not in dominant_id_to_topics:
				dominant_id_to_topics[max_topic_id] = []

			dominant_id_to_topics[max_topic_id].append(topics)

		dominant_id_to_cum_topics = {}
		for k, v in dominant_id_to_topics.iteritems():
			dominant_id_to_cum_topics[k] = np.mean(np.array(v), axis=0) * 100. 

		df = pd.DataFrame.from_dict(dominant_id_to_cum_topics)
		df.columns = [get_topic_label(x) for x in df.columns.values]
		df.index = [get_topic_label(x) for x in df.index.values]
		df['max'] = 0.
		new_index = []

		for index, row in df.iterrows():
			df['max'][index] = max(row)
			df[index][index] = 0.0
			new_index.append('{} ({}%)'.format(index, round(max(row), 2)))
		df.index = new_index
		df = df.sort_values(by=['max'], ascending=False)
		df = df.drop(['max'], axis=1)
		df = df.reindex(sorted(df.columns), axis=1)
		ax = sns.heatmap(df, cmap = "Blues", annot = True, vmin = 0., vmax = 10., square = True, annot_kws = {"size": 11},
							fmt = '.1f', mask= df <= 0.0, linewidths = .5, cbar = False, yticklabels=True)
		ax.xaxis.tick_top()
		plt.yticks(rotation=0)
		plt.xticks(rotation=90, ha = 'left') 
		fig = ax.get_figure()
		fig.set_size_inches(19, 6)

		fig.savefig(os.path.join(self.plot_save_folder,plot_save_name), bbox_inches='tight')


	def plot_topics_in_journals(self, plot_save_name = 'topics-in-journals.pdf'):
		journal_to_topics = {}
		D = self.db.read_collection(collection = 'publications')
		for i, d in enumerate(D):
			if i % 1000 == 0: print('Processing document {}/{}'.format(i, D.count()))
			journal = d['journal']
			if d.get('topics') is not None:
				if journal not in journal_to_topics:
					journal_to_topics[journal] = []
				topics = [value for key, value in sorted(d['topics'].iteritems(), key=lambda x: int(x[0]))]
				journal_to_topics[journal].append(topics)

		journal_to_cum_topics = get_journal_to_cum_topics(journal_to_topics)
		df = pd.DataFrame.from_dict(journal_to_cum_topics).T
		df.columns = [get_topic_label(x) for x in df.columns.values]
		ax = sns.heatmap(df, cmap = "Blues", annot = True, vmin = 0., vmax = .3, square = True, annot_kws = {"size": 11}, fmt = '.2f', mask= df <= 0.0, linewidths = .5, cbar = False, yticklabels = True)

		ax.xaxis.tick_top()
		plt.yticks(rotation = 0)
		plt.xticks(rotation = 90, ha = 'left') 
		fig = ax.get_figure()
		fig.set_size_inches(10, 10)
		fig.savefig(os.path.join(self.plot_save_folder, plot_save_name), bbox_inches='tight')
		plt.close()

def get_year_to_topics(D):
	year_to_topics = {}

	for d in D:
		if int(d['year']) not in year_to_topics:
			year_to_topics[int(d['year'])] = []

		topics = [value for key, value in sorted(d['topics'].iteritems(), key=lambda x: int(x[0]))]
		year_to_topics[int(d['year'])].append(topics)

	return year_to_topics


def get_year_to_cum_topics(year_to_topics):
	year_to_cum_topics = {}
	for k, v in year_to_topics.iteritems():
		mean_topics = np.mean(np.array(v), axis = 0)
		year_to_cum_topics[k] = mean_topics
	return year_to_cum_topics

def get_journal_to_cum_topics(journal_to_topics):
	journal_to_cum_topics = {}
	for k, v in journal_to_topics.iteritems():
		mean_topics = np.mean(np.array(v), axis = 0)
		journal_to_cum_topics[k] = mean_topics
	return journal_to_cum_topics

