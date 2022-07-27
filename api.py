from flask import Flask, send_file, request, jsonify
from flask_cors import CORS, cross_origin
from keras.models import load_model
from bson.objectid import ObjectId
from PIL import Image, ImageFile
from skimage import transform
from flask import Response
import tensorflow as tf
import pandas as pd
import urllib.parse
import numpy as np
import subprocess
import datetime
import pymongo
import base64
import random
import json
import os
import re


ImageFile.LOAD_TRUNCATED_IMAGES = True


app = Flask(__name__)
app.config["DEBUG"] = True

cors = CORS(app, resources={r"/api/*": {"origins": "*",}})
cors = CORS(app, resources={r"/static/*": {"origins": "*",}})


model=None
db=None
annotations=None

animals_map={
"sangliers":["tayassu pecari","pecari tajacu", "phacochoerus africanus", "potamochoerus larvatus", "sus scrofa"],
"rongeurs":["cuniculus paca", "didelphis marsupialis", "dasyprocta punctata", "sylvilagus brasiliensis",
"philander opossum","lepus saxatili", "herpestes sanguineus","cricetomys gambianus","ichneumia albicauda","xerus erythropus",
"helogale parvula","hystrix brachyura","lariscus insignis","hystrix cristata","procavia capensis","dasyprocta fuliginosa",
"myoprocta pratti","proechimys sp","didelphis sp","xerus rutilus","atherurus africanus","funisciurus carruthersi","protoxerus stangeri",
"paraxerus boehmi","oenomys hypoxanthus","hybomys univittatus","colomys goslingi","hylomyscus stella","mus minutoides","praomys tullbergi",
"malacomys longipes","deomys ferrugineus","funisciurus pyrropus","thryonomys swinderianus","anomalurus derbianus"],
"chevres":["mazama americana","mazama gouazoubira","capra aegagrus","tragelaphus scriptus","raphicerus campestris","aepyceros melampus",
"tragelaphus oryx","kobus ellipsiprymnus","alcelaphus buselaphus","madoqua guentheri","nanger granti","eudorcas thomsonii","oryx beisa",
"muntiacus muntjak","tragelaphus strepsiceros", "capricornis sumatraensis"],
"vaches":["bos taurus","syncerus caffer","cephalophus silvicultor"],
"moutons":["ovis aries","equus africanus","equus ferus"],
"oiseaux":["geotrygon montana","penelope jacquacu","aramides cajaneus","tinamus major","crypturellus sp","tigrisoma lineatum","turtur calcospilos",
"eupodotis senegalensis","lophotis gindiana","chalcophaps indica","streptopilia senegalensis","crypturellus soui","momotus momota","geotrygon sp",
"penelope purpurascens","brotogeris sp","acryllium vulturinum","tockus deckeni","rollulus rouloul","lophura inornata","polyplectron chalcurum",
"motacilla flava","andropadus latirostris","andropadus virens"],
"singes":["chlorocebus pygerythrus","macaca nemestrina","macaca fascicularis","pan troglodytes","cercopithecus mitis"],
"autres":["unknown","human","empty","felis silvestris","unknown bird","leptotila rufaxilla","unknown bat","unidentifiable","unknown squirrel"]}

def init_app() :

    print("Démarrage de cultura...")

    atlas_username=urllib.parse.quote_plus("cultura_user")
    atlas_pwd=urllib.parse.quote_plus("p@sser123") #TODO : cacher le mot de passe dans un fichier .env

    url="mongodb+srv://{0}:{1}@cultura.mbcjy.mongodb.net/?retryWrites=true&w=majority".format(atlas_username,atlas_pwd)
    client=pymongo.MongoClient(url)
    _db=client["cultura"]
    if _db is not None:
        print("connected to db.")

    p=None

    if app.config["DEBUG"]==False:
        p=subprocess.Popen(['python3', 'motion_detection.py'])
    else :
        p=subprocess.Popen(['python3', 'motion_detection.py', "--video", "./metadata/test_videos/vaches.mp4"])
        #insérer 10 données de tests
        _col=_db["cultura_stats"]

        pseudo_especies=['vaches','singes','chevres','oiseaux']
        print("creation de données aléatoires pour les statistiques.")
        _col.delete_many({})
        for i in range(10):
            _id= ObjectId()
            animal=random.choice(pseudo_especies)
            if i%3==0 :
                state="pas d'invasion"
            else:
                if animal=="oiseaux":
                    state="invasion d'"+animal
                else:
                    state="invasion de "+ animal
            pseudo_date=datetime.datetime(2020, 7, i+10,i+2,i+30,i+12)
            record={"_id":_id,"date":pseudo_date,"etat":state}
            _col.insert_one(record)


    print("surveillance en temps réel en cours...")

    model = load_model('cultura_model.h5')

    with open('./metadata/annotations.json', encoding='utf-8') as json_file:
        annotations =json.load(json_file)

    return _db,model,annotations


@app.after_request
def after_request(response):
    response.headers.add('Accept-Ranges', 'bytes')
    return response


#le status actuel du champ
#enregistrer sur la bd si la catégorie est menaçante
@app.route('/api/get_status',methods=['GET'])
def get_current_status():
    query_parameters = request.args
    q = query_parameters.get('q',None)

    image = Image.open("./metadata/detected_images/detected_img.jpg")
    np_image = np.array(image).astype('float32')/255
    np_image = transform.resize(np_image, (64, 64, 3))
    np_image = np.expand_dims(np_image, axis=0)
    pred=model.predict(np_image)
    pred_category = [np.argmax(i) for i in pred][0]
    response=None
    df_categories = pd.DataFrame.from_records(annotations["categories"])
    espece_name=df_categories[df_categories["id"]==pred_category].name
    espece_name=espece_name.to_numpy()[0]
    is_invasion=False
    animal_name=""
    for key, value in animals_map.items():
        if espece_name in value:
            is_invasion=True
            animal_name=key
            break;

    if q is not None :
        if q=="status" :
            if is_invasion==True:
                response=jsonify({'status':"alerte invasion",'message':'Alerte!!! des '+ animal_name+' rodent près de vos cultures, agissez avant qu\'ils ne fassent des ravages.'})
                response.headers.add('Access-Control-Allow-Origin', '*')
                return response
                #TODO send informations to the database
            else:
                response=jsonify({'status':"no invasion",'message':'Surveillance de votre champ en cours. Il n\'y a actuellement rien à signaler sur votre champ.'})
                response.headers.add('Access-Control-Allow-Origin', '*')
                return response
        else :
            return "bad request",400


@app.route('/static/get_current_status_image')
def get_image():

    filename = './metadata/detected_images/detected_img.jpg'
    return send_file(filename, mimetype='image/jpeg')


#les statistiques de la journée / semaine
#récupérer les 10 dernières données de la même journée
#récupérer les 10 dernières données de la semaine
@app.route('/api/get_stats',methods=['GET'])
def get_stats():
    query_parameters = request.args
    how = query_parameters.get('how',None)

#TODO: changer la requête de sorte à récupérer selon le temps
    if how is not None :
        if how=="daily" :
            return jsonify({'status':'...'})
        elif how=="weekly" : 
            result= list(db["cultura_stats"].find(sort=[( '_id', pymongo.DESCENDING )]).limit(10))
            for record in result :
                record["_id"]=str(record["_id"])

            response=jsonify({'stats':result})
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response

        else :
            return "bad request",400



#connexion à la plateforme
#TODO : create serializer to valid incomming data
@app.route('/api/login',methods=['POST'])
@cross_origin()
def login():
    db_collection=db["cultura_users"]

    data=request.json
    username=data['username']
    password=data['password']

    query = { "username": username, "password":password}
    user=db_collection.find_one(query)

    if user is not None :
        return jsonify({'username':username})
    else:
        return "utilisateur non existant",404



#visualiser l'enregistrement en cours

def get_chunk(byte1=None, byte2=None):
    full_path = "./metadata/saved_vid.mp4"
    file_size = os.stat(full_path).st_size
    start = 0
    
    if byte1 < file_size:
        start = byte1
    if byte2:
        length = byte2 + 1 - byte1
    else:
        length = file_size - start

    with open(full_path, 'rb') as f:
        f.seek(start)
        chunk = f.read(length)
    return chunk, start, length, file_size



@app.route('/api/video')
def get_video():
    range_header = request.headers.get('Range', None)
    byte1, byte2 = 0, None
    if range_header:
        match = re.search(r'(\d+)-(\d*)', range_header)
        groups = match.groups()

        if groups[0]:
            byte1 = int(groups[0])
        if groups[1]:
            byte2 = int(groups[1])
       
    chunk, start, length, file_size = get_chunk(byte1, byte2)
    response = Response(chunk, 206, mimetype='video/mp4',
                      content_type='video/mp4', direct_passthrough=True)
    response.headers.add('Content-Range', 'bytes {0}-{1}/{2}'.format(start, start + length - 1, file_size))
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


#effectuer une action de défense du champ en cas d'alerte

@app.route('/api/defense_action',methods=['POST'])
def defend_domain():
    query_parameters = request.args
    how = query_parameters.get('how',None)
    request.form.get('something')

    if how is not None :
        if how=="send_sound_waves" :
            return jsonify({'status':'En cours de développement.'})
        elif how=="alert_nearby_services" : 
            return jsonify({'status':'En cours de développement.'})
        else :
            return "bad request",400


if __name__ == '__main__':
    db,model,annotations=init_app()
    app.run(host='0.0.0.0', port=8081,threaded=True)