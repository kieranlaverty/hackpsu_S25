import pymongo
from pymongo import MongoClient
from gridfs import GridFS
import datetime
import key.api

def connect_to_mongodb():
    db = getdb()
    fs = GridFS(db)
    return

def getdb():
    client = pymongo.MongoClient(key.api.mongoURL())
    db = client["pi_camera"]
    return db

def getCollection(name):
    db = getdb()
    collection = db[name]
    return collection.find({})

def storeImage(image_path):
    day = datetime.now().strtime("%d")
    db = getdb()
    collection = db[day]

    return 
    
def retrieveDoc(collection, doc):
    db = getdb()


    return
def loadImage(Path, collection, day):

    return


getdb()
