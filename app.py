#app
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import tensorflow as tf
from flask import Flask, request, render_template
from keras.preprocessing.image import ImageDataGenerator
import os
from os import remove
app = Flask(__name__, template_folder='templates',static_folder='static')

# refere this code for any errors while directory change and all gowda
#train = pd.read_csv("C:\\Users\\DharaniPrasadS\\Desktop\\flaskapp\\dataset\\insat_3d_ds - Sheet.csv")
#loaded_model= tf.keras.models.load_model('C:\\Users\\DharaniPrasadS\\Desktop\\flaskapp\\model\\Model.h5')
#train_datagen = ImageDataGenerator(rescale=1.0/255.0)
#train_data1 = train_datagen.flow_from_dataframe(train,directory="C:\\Users\\DharaniPrasadS\\Desktop\\flaskapp\\uploads",subset="training",x_col="img_name",y_col="label",target_size=(512, 512),batch_size=16,class_mode='raw')
#train = pd.read_csv("C:\\Users\\DharaniPrasadS\\Desktop\\flaskapp\\dataset\\insat_3d_ds - Sheet.csv")
#loaded_model= tf.keras.models.load_model('C:\\Users\\DharaniPrasadS\\Desktop\\flaskapp\\model\\Model.h5')
#train_datagen = ImageDataGenerator(rescale=1.0/255.0)
#train_data1 = train_datagen.flow_from_dataframe(train,directory="C:\\Users\\DharaniPrasadS\\Desktop\\flaskapp\\uploads",subset="training",
#                                              x_col="img_name",y_col="label",target_size=(512, 512),batch_size=16,class_mode='raw')

#code to help in de-buugg gowda
#print(train_data1)
#predict1=loaded_model.predict(train_data1 , verbose=1).round(2)
#print(predict1[0][0])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['GET','POST'])
def submit():
    if request.method == 'POST':
        global file
        file = request.files['file']
        print(file.filename)
        file.save('C:\\Users\\DharaniPrasadS\\Desktop\\flaskapp\\uploads\\'+file.filename)
        train = pd.read_csv("C:\\Users\\DharaniPrasadS\\Desktop\\flaskapp\\dataset\\insat_3d_ds - Sheet.csv")
        global loaded_model
        loaded_model= tf.keras.models.load_model('C:\\Users\\DharaniPrasadS\\Desktop\\flaskapp\\model\\Model.h5')
        train_datagen = ImageDataGenerator(rescale=1.0/255.0)
        global train_data1
        train_data1 = train_datagen.flow_from_dataframe(train,directory="C:\\Users\\DharaniPrasadS\\Desktop\\flaskapp\\uploads",subset="training",
                                                x_col="img_name",y_col="label",target_size=(512, 512),batch_size=16,class_mode='raw')
        return render_template('index.html')
   
@app.route('/predict',methods=['GET','POST'])
def predict():
    train = pd.read_csv("C:\\Users\\DharaniPrasadS\\Desktop\\flaskapp\\dataset\\insat_3d_ds - Sheet.csv")
    loaded_model= tf.keras.models.load_model('C:\\Users\\DharaniPrasadS\\Desktop\\flaskapp\\model\\Model.h5')
    train_datagen = ImageDataGenerator(rescale=1.0/255.0)
    train_data1 = train_datagen.flow_from_dataframe(train,directory="C:\\Users\\DharaniPrasadS\\Desktop\\flaskapp\\uploads",subset="training",
                                                x_col="img_name",y_col="label",target_size=(512, 512),batch_size=16,class_mode='raw')
    print(train_data1)
    predict1=loaded_model.predict(train_data1, verbose=1)
    print(int(predict1[0][0]))
    x=''
    if (0<=predict1[0][0]<= 28):
        x="Depression"
    if (28<=predict1[0][0]<=34):
        x="Deep Depression"
    if (34<=predict1[0][0]<=48):
        x="Cyclonic Storm"
    if (48<=predict1[0][0]<=64):
        x="Severe Cyclonic Storm"
    if (64<=predict1[0][0]<=120):
        x="Very Severe Cyclonic Storm"
    if (120<=predict1[0][0]):
        x="Super Cyclonic Storm"
    print(x)
    return render_template('index.html', prediction_text = predict1[0][0],knot_value=x)

@app.route('/reload', methods=['GET','POST'])
def reload():
    path = "C:\\Users\\DharaniPrasadS\\Desktop\\flaskapp\\uploads\\"
    dir_list = os.listdir(path)
    location= path+str(dir_list[0])
    os.remove(location)
    return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

    