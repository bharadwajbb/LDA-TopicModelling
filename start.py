from extraction import Extraction
from preprocessing import Preprocessing
from transformation import Transformation
from datamining import Datamining
from evaluation import Evaluation
from interpretation import Interpretation
from helper_functions import *

from datetime import datetime


EXTRACTION = False
PREPROCESSING = True
TRANSFORMATION = False
DATAMINING = False
EVALUATION = False
INTERPRETATION = False


if __name__ == "__main__":

	if EXTRACTION:
		extraction = Extraction()
		extraction.extract_publications()


	if PREPROCESSING:
		preprocessing = Preprocessing()
		preprocessing.full_text_preprocessing()
		preprocessing.general_preprocessing()


	if TRANSFORMATION:
		transformation = Transformation()
		transformation.transform_for_lda()


	if DATAMINING:
		datamining = Datamining()
		datamining.execute_lda()


	if EVALUATION:
		evaluation = Evaluation()
		evaluation.calculate_coherence()
		evaluation.plot_coherence()
		evaluation.output_lda_topics()


	if INTERPRETATION:
		interpretation = Interpretation()
		interpretation.infer_document_topic_distribution()
		interpretation.get_document_title_per_topic()
		interpretation.plot_topics_over_time()
		interpretation.plot_topics_over_time_stacked()
		interpretation.plot_topic_co_occurrence()
		interpretation.plot_topics_in_journals()
