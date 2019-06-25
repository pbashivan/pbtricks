from __future__ import print_function
from pymongo import MongoClient

from functools import wraps
from pymongo.errors import ServerSelectionTimeoutError
import subprocess

DEFAULT_LOCAL_PORT_NUM = 30001
DEFAULT_REMOTE_PORT_NUM = 27017


def _check_connection(f):
  @wraps(f)
  def wrapper_func(self, *args, **kwargs):
    try:
      self._db.collection_names()
      return f(self, *args, **kwargs)
    except ServerSelectionTimeoutError:
      print('connection is down, trying to reconnect...')
      subprocess.Popen(['ssh', '-N', '-L', f'{DEFAULT_LOCAL_PORT_NUM}:localhost:{DEFAULT_REMOTE_PORT_NUM}',
                        'bashivan@braintree-cpu-1.mit.edu'])
      return f(self, *args, **kwargs)
  return wrapper_func


class MongoInterface(object):
  def __init__(self, server, port, db_name, collection_name):
    self._client = MongoClient(server, port)
    self._db = self._client[db_name]
    self._collection = self._db[collection_name]

  @_check_connection
  def entry_exists(self, entry):
    count = self._collection.count(entry)
    if count > 0:
      return True
    else:
      return False

  def insert_entry(self, entry):
    """
    Inserts an entry in the table.
    :param entry: (dict) a dictionary containing model definition and other info
    :return:
    """
    if self.entry_exists(entry):
      return -1
    else:
      self._collection.insert_one(entry)
      return 0

  @_check_connection
  def find_all_entries(self, entry):
    """
    Looks for matching entry for a model_def in the table.
    :param entry: (DbEntry) entry object for inquiry
    :return: Returns the entry if exists otherwise returns None
    """
    return self._collection.find(entry)

  @_check_connection
  def find_entry(self, entry):
    """
    Looks for matching entry for a model_def in the table.
    :param entry: (DbEntry) entry object for inquiry
    :return: Returns the entry if exists otherwise returns None
    """
    return self._collection.find_one(entry)

  @_check_connection
  def update_entry(self, criteria, entry, upsert=False):
    """
    Updates the entry in DB with values in entry.
    :param criteria: (dict) a dictionary describing the criteria for filtering the entries.
    :param entry: (dict) A dictionary containing keys and values to be added to the entry.
    :param upsert: (bool) whether to upsert or not
    :return:
    """
    if self.entry_exists(criteria):
      self._collection.update(criteria,
                              {'$set': entry}, upsert=False)
    else:
      if upsert:
        self._collection.update(criteria,
                                {'$set': entry}, upsert=True)
      else:
        raise MissingEntry()

  def close(self):
    """
    closes the connection to MongoDB
    :return:
    """
    self._client.close()


class MissingEntry(Exception):
  """Entry does not exist in MongoDB.
  """
