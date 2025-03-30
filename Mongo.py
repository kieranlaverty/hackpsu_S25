import pymongo
from pymongo import MongoClient
from gridfs import GridFS
import datetime
from key.api  import mongoURL
import os
from PIL import Image
import io
from bson.binary import Binary

def connect_to_mongodb():
    db = getdb()
    fs = GridFS(db)
    return fs, db

def getdb(name):
    client = pymongo.MongoClient(mongoURL())
    db = client[name]
    return db

def getCollection(collection_name, db_name):
    db = getdb(db_name)
    collection = db[collection_name]
    return collection.find({})

def insert_image():
    client = pymongo.MongoClient(mongoURL)
    db = client['mydatabase']
    collection = db['images']

    directory = "C:\Users\Owner\Pictures\Screenshots\Screenshot"
    image_filename = "2025-03-29 234628.png"

    image_path = os.path.join(directory, image_filename)

    image = Image.open(image_path)

    image.show()

    image_document = {
        'filename': 'image.jpg',
        'data': image_data
    }
    collection.insert_one(image_document)

def read_image():
    client = pymongo.MongoClient(mongoURL)
    db = client['mydatabase']
    collection = db['images']
    image_document = collection.find_one({'filename': 'image.jpg'})

    image_data =image_document['data']
    image = Image.open(io.BytesIO(image_data))
    image.show()

def storeImage(image_path, db_name="pi_camera"):
    day = datetime.datetime.now().strftime("%d")
    db = getdb(db_name)
    collection = db[day]

    return 
    
def retrieveDoc(collection, doc):
    db = getdb()


    return
def loadImage(Path, collection, day):

    return


insert_image()
read_image()
