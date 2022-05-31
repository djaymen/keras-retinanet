# Code adapted from https://github.com/fizyr/keras-retinanet
# jasperebrown@gmail.com
# 2020

# This script loads a single image, runs inferencing on it
# and saves that image back out with detections overalaid.

# You need to set the model_path and image_path below

# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image

# import miscellaneous modules
import cv2
import pandas as pd
import os
import numpy as np
import time
from PIL import Image
from tqdm import tqdm
from pathlib import Path
# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

model_path = '/content/Retinanet/RetinanetModels/PlumsInference.h5'
directory = '/content/save'
csv_file='/content/detection.csv'
confidence_cutoff = 0.5 #Detections below this confidence will be ignored

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
#keras.backend.tensorflow_backend.set_session(get_session())

# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
#model_path = os.path.join('..', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')



# load retinanet model
print("Loading Model: {}".format(model_path))
model = models.load_model(model_path, backbone_name='resnet50')

#Check that it's been converted to an inference model
try:
    model = models.convert_model(model)
except:
    print("Model is likely already an inference model")

# load label to names mapping for visualization purposes

list_files=[]
results=[]
final=[]
l_size=[224,250,275,300,316,416]
for size in l_size:
  image_dir=directory+str(size)+'/'
  list_files=[]
  results=[]
  for file in os.listdir(image_dir):list_files.append(file)
  for i in tqdm(range(len(list_files))):
    image_path=image_dir+list_files[i] 
    name=Path(image_path).stem   
    lx = name.split("_")
    image = np.asarray(Image.open(image_path).convert('RGB'))
    image = image[:, :, ::-1].copy()
    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    # Image formatting specific to Retinanet
    image = preprocess_image(image)
    image, scale = resize_image(image)
    
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    
    # correct for image scale
    boxes /= scale

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
       # scores are sorted so we can break
       if score < confidence_cutoff:
         break
         
       b = np.array(box).astype(int)
       results.append((int(lx[0]),int(lx[1]),b[0], b[1],b[2], b[3]))
  
       #cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)
  df = pd.DataFrame(results)
  csv_file = '/content/detection'+str(size)+'.csv' 
  df.to_csv(csv_file,index=False)      
  print("successfully detected......",size)  
#df = pd.DataFrame(results)  
#df.to_csv(csv_file,index=False)  
