import sys, re
from helper_functions import *

class Extraction():

	def extract_publications(self, save_folder = os.path.join('files', 'pdf')):
		journal = 'NIPS'
		domain = 'https://papers.nips.cc'
		p1_content = return_html(domain).text
		p1_links = re.findall(r'<a href="(/paper_files.*?)">', p1_content) 
		for l in p1_links:
			year = l[-4:]
			l2 = '{}{}'.format(domain, l)
			p2_content = return_html(l2).text
			matches = re.findall(r'<a title="paper title" href="(.*?)">(.*?)</a>', p2_content)
			processed_publications = [x.split(os.sep)[-1][0:-4] for x in read_directory(os.path.join(save_folder, journal, year))]
			for i, (link,title) in enumerate(matches):
				title = ''.join(e for e in title if e.isalnum())

				if title in processed_publications:
					continue
				modified_link = link.replace("hash","file")
				modified_link = modified_link.replace("Abstract","Paper")
				modified_link = modified_link.replace("html", "pdf")
				try:
					pdf_link = '{}{}'.format(domain, modified_link)
					pdf_name = title
					save_pdf(pdf_link, 
								folder = os.path.join(save_folder, journal, year), 
								name = pdf_name + ".pdf",
								overwrite = False)
				except (Exception):
					continue


