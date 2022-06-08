# Code adapted from https://github.com/fizyr/keras-retinanet
# jasperebrown@gmail.com
# 2020

# This script loads a single image, runs inferencing on it
# and saves that image back out with detections overalaid.

# You need to set the model_path and image_path below

# import keras
import keras
import math
import shutil

import pandas as pd
# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image

import argparse

PARSER = argparse.ArgumentParser(usage="""
             [options]""")

PARSER.add_argument('-img_path', '--img_path', type=str, required = True,
                    help = """Image path""")
                   
PARSER.add_argument('-model_path', '--model_path', type=str, required = True,
                    help = """Inference model path""")                    

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


def contains(bbox,list_prep):
  for box in list_prep:
    if (bbox[0]<box[0] and bbox[1]<box[1] and bbox[2]>box[2] and bbox[3]>box[3]):
     
      return False
  return True 


def color_rec(img,rec,color):
  x,y,h,w,_=rec
  
  for i in range(y,w):
    for j in range(x,h):
      
      l,m,r=img[i][j]
      
      if(l<190 and m<190 and m<190):
         img[i][j]=np.array([color[0],color[1],color[2]])
  return img 


def border_contour(gray,depth=20):
  
  l=len(gray)
  c=len(gray[0])
  #left border
  for i in range(l):
    for j in range(depth):
      gray[i][j]=255
  #right border
  for i in range(l):
    for j in range(c-depth,c):
      gray[i][j]=255

  #up border
  for j in range(c):
    for i in range(depth):
      gray[i][j]=255
  #bottom border
  for j in range(l):
    for i in range(l-depth,l):
      gray[i][j]=255
  
  return gray

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


if __name__ == '__main__':
   args = PARSER.parse_args()
   #model_path = '/content/Retinanet/RetinanetModels/PlumsInference.h5'
   model_path=args.model_path

   Path("./save224").mkdir(parents=True, exist_ok=True)
   Path("./save250").mkdir(parents=True, exist_ok=True)
   Path("./save275").mkdir(parents=True, exist_ok=True)
   Path("./save300").mkdir(parents=True, exist_ok=True)
   Path("./save316").mkdir(parents=True, exist_ok=True)


   image_path=args.img_path
   test=cv2.imread(image_path)
   gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
   im =border_contour(gray)
   l_size=[224,250,275,300,316]
   for size in l_size:
     M=size
     N=size
     tiles = [im[x:x+M,y:y+N] for x in range(0,im.shape[0],M) for y in range(0,im.shape[1],N)]
     index=0
     id=0
     dev=math.ceil(test.shape[1]/size)
     for i in range(len(tiles)): 
      if(i% dev==0):
       index+=1
       id=0
      im_path='./save'+str(size)+"/"+str(index)+'_'+str(id)+'.jpg'
      id+=1
      cv2.imwrite(im_path,tiles[i])   




   directory = './save'
   csv_file='./detection.csv'
   confidence_cutoff = 0.5 #Detections below this confidence will be ignored  
   
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
   #l_size=[224,250,275,300,316]
   for size in l_size:
    image_dir=directory+str(size)+'/'
    list_files=[]
    #results=[]
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

      for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < confidence_cutoff:
          break
         
        b = np.array(box).astype(int)
        results.append((size,int(lx[0]),int(lx[1]),b[0], b[1],b[2], b[3],label))
      
    print("successfully detected......",size)  

   df = pd.DataFrame(results)
   df.to_csv(csv_file,index=False)
   """
   #df=pd.read_csv(csv_file)
   #arr = df.to_numpy()
   list_prep=[]
   for row in results:
     i=(row[1]-1)*row[0]
     j=row[2]*row[0]
     list_prep.append((j+row[3], i+row[4], j+row[5], i+row[6],labels_to_names[row[7]])) 
   list_final=[]
   for bbox in list_prep:
     if(contains(bbox,list_prep)==True):
       list_final.append(bbox)

   image=cv2.imread(image_path)
   for row in list_final:
     color = (2, 255, 2)
     thickness = 1
     if row[4].startswith('arrow'):
       image=color_rec(image,row,(255, 2, 77))
     elif(row[4].startswith('dotted')):  
       image=color_rec(image,row,(55, 2, 245))  
     elif(row[4].startswith('o_sign')):  
       image=color_rec(image,row,(20, 150, 245)) 
     elif(row[4].startswith('double_slash')):  
       image=color_rec(image,row,(95, 230, 111))     
     else:
        cv2.rectangle(image, (row[0], row[1]), (row[2], row[3]), color, thickness, cv2.LINE_AA) 
   """
   shutil.rmtree("./save224") 
   shutil.rmtree("./save250") 
   shutil.rmtree("./save275") 
   shutil.rmtree("./save300") 
   shutil.rmtree("./save316")
   #shutil.rmtree('./detection.csv')       
   #cv2.imwrite('./image_det.jpg',image)   
   
