from extraction import Extraction
from preprocessing import Preprocessing
from transformation import Transformation
from datamining import Datamining
from evaluation import Evaluation
from interpretation import Interpretation
from helper_functions import *

from datetime import datetime


if __name__ == "__main__":
	extraction = Extraction()
	extraction.extract_publications()

	preprocessing = Preprocessing()
	preprocessing.full_text_preprocessing()
	preprocessing.general_preprocessing()


	transformation = Transformation()
	transformation.transform_for_lda()


	datamining = Datamining()
	datamining.execute_lda()



	evaluation = Evaluation()
	evaluation.calculate_coherence()
	evaluation.plot_coherence()
	evaluation.output_lda_topics()


	interpretation = Interpretation()
	interpretation.infer_document_topic_distribution()
	interpretation.get_document_title_per_topic()
	interpretation.plot_topics_over_time()
	interpretation.plot_topic_co_occurrence()
	interpretation.plot_topics_in_journals()
