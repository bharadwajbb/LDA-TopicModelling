
from pymongo import MongoClient
import time, sys
from bson.objectid import ObjectId


class MongoDatabase:
	def __init__(self, client = 'LDA-github-2', host='localhost'):
		self.client = MongoClient(port=27017, host=host, connect=True)
		self.db = self.client[client]


	def read_collection(self, collection):
		with self.client.start_session() as session:
			try:
				cursor = self.db[collection].find({}, session=session, no_cursor_timeout=True)
				return list(cursor)
			except (Exception):
				print("Exception Raised")
				exit(1)

	def insert_one_to_collection(self, collection, doc):
		try:
			self.db[collection].insert_one(doc)
		except (Exception):
			exit(1)


	def update_collection(self, collection, doc):
		try:	
			self.db[collection].update({'_id' : ObjectId(doc['_id'])},
									doc
									,upsert = False)
		except (Exception):
			exit(1)
