import sys, re
from gensim import corpora, models
from helper_functions import *


class Datamining():
	def execute_lda(self, file_folder = os.path.join('files', 'lda'), save_folder = os.path.join('files', 'models')):
		K = range(2,20 + 1)
		dir_priors = ['auto']
		random_states = [42,99]
		num_passes = [5,10,15,20]
		iterations = [200]

		dictionary, corpus = get_dic_corpus(file_folder)

		for k in K:
			for dir_prior in dir_priors:
				for random_state in random_states:
					for num_pass in num_passes:
						for iteration in iterations:
							target_folder = os.path.join(save_folder, str(k), dir_prior, str(random_state), str(num_pass), str(iteration))
							if not (os.path.exists(target_folder)):
								create_directory(target_folder)

								model = models.LdaModel(corpus, 
														id2word = dictionary,
														num_topics = k,
														iterations= iteration, 
														passes = num_pass, 
														minimum_probability = 0, 
														alpha = dir_prior, 
														eta = dir_prior, 
														eval_every = None,
														random_state= random_state)

								model.save(os.path.join(target_folder, 'lda.model'))
								
							else:
								print('LDA model already exists, skipping ...')
