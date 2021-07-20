import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import cv2
import tensorflow as tf

# loading the database
base = np.load('/content/drive/My Drive/cullpdb+profile_6133.npy')
base = np.reshape(base, (-1, 700, 57))
a = np.arange(0,21)
b = np.arange(35,56)
c = np.hstack((a,b))
previsores = base[:, :, b]
classes = base[:, :, 22:30]

# functions to create the image
def timeseries_to_classification(sequence):
    datas = []
    for i in range(len(sequence)):
        for j in range(21):
            datas.append(sequence[i, j])
    return np.array(datas)

def dados(window, start, end, leng=8, features=21):
    
    data = []
    label = []
    
    for i in range(start, end):
        protein = previsores[i]
        ss = classes[i]
        
        x = protein[~np.all(protein == 0, axis=1)]
        y = ss[~np.all(ss == 0, axis=1)]
        
        padding = np.zeros(window * features).reshape(window, features)
        x = np.vstack((padding, x))
        x = np.vstack((x, padding))
        
        padding = np.zeros(window * leng).reshape(window, leng)
        y = np.vstack((padding, y))
        y = np.vstack((y, padding))
        
        cont = (window * 2) + 1
        
        for i in range(x.shape[0] - (window * 2)):
            data.append(timeseries_to_classification(x[i:cont]))
            label.append(y[i+window:i+window+1])
            cont += 1
    
    data = np.array(data)
    label = np.array(label)
    
    return data, np.reshape(label, (label.shape[0], -1))

os.mkdir('CB6133')
os.mkdir('CB6133/Train')
os.mkdir('CB6133/Val')
os.mkdir('CB6133/Test')


# generating image - Train
cont_img = 0
img_name = []
img_target = []

for idx in tqdm(range(1, 5601)):
  img, labels = dados(10, idx-1, idx)
  img = np.reshape(img, (-1, 21, 21, 1))
  labels = np.argmax(labels, axis=1)

  for index in range(len(img)):
    a = cv2.normalize(img[index], None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    a = a.astype(np.uint8)
    a = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
    a = cv2.resize(a, (63, 63))
    cv2.imwrite('/content/CB6133/Train/train_img'+ str(cont_img) + '.png', a)
    img_name.append('train_img'+ str(cont_img) + '.png')
    img_target.append(labels[index])
    cont_img += 1

dic = {'Id': img_name, 'Target': img_target}
df = pd.DataFrame(dic)
df.to_csv('/content/CB6133/train.csv', index=False)



# generating image - Val
cont_img = 0
img_name = []
img_target = []

for idx in tqdm(range(5878, 6134)):
  img, labels = dados(10, idx-1, idx)
  img = np.reshape(img, (-1, 21, 21, 1))
  labels = np.argmax(labels, axis=1)

  for index in range(len(img)):
    a = cv2.normalize(img[index], None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    a = a.astype(np.uint8)
    a = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
    a = cv2.resize(a, (63, 63))
    cv2.imwrite('/content/CB6133/Val/val_img'+ str(cont_img) + '.png', a)
    img_name.append('val_img'+ str(cont_img) + '.png')
    img_target.append(labels[index])
    cont_img += 1

dic = {'Id': img_name, 'Target': img_target}
df = pd.DataFrame(dic)
df.to_csv('/content/CB6133/val.csv', index=False)



# generate image - Test
cont_img = 0
img_name = []
img_target = []

for idx in tqdm(range(5606, 5878)):
  img, labels = dados(10, idx-1, idx)
  img = np.reshape(img, (-1, 21, 21, 1))
  labels = np.argmax(labels, axis=1)

  for index in range(len(img)):
    a = cv2.normalize(img[index], None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    a = a.astype(np.uint8)
    a = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
    a = cv2.resize(a, (63, 63))
    cv2.imwrite('/content/CB6133/Test/test_img'+ str(cont_img) + '.png', a)
    img_name.append('teste_img'+ str(cont_img) + '.png')
    img_target.append(labels[index])
    cont_img += 1

dic = {'Id': img_name, 'Target': img_target}
df = pd.DataFrame(dic)
df.to_csv('/content/CB6133/test.csv', index=False)



# loading dataframes with images path
df_train = pd.read_csv('/content/CB6133/train.csv')
df_val = pd.read_csv('/content/CB6133/val.csv')
df_train['Target'] = df_train['Target'].astype('str')
df_val['Target'] = df_val['Target'].astype('str')
df_teste = pd.read_csv('/content/CB6133/test.csv')
df_teste['Target'] = df_teste['Target'].astype('str')

datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input)

train_gen = datagen.flow_from_dataframe(
      df_train, directory='/content/CB6133/Train', x_col='Id', y_col='Target',
      target_size=(63, 63), class_mode='categorical', batch_size=32, shuffle=True, seed=42)

val_gen = datagen.flow_from_dataframe(
      df_val, directory='/content/CB6133/Val', x_col='Id', y_col='Target',
      target_size=(63, 63), class_mode='categorical', batch_size=32, shuffle=False, seed=42)

test_gen = datagen.flow_from_dataframe(
      df_teste, directory='/content/CB6133/Test', x_col='Id', y_col='Target',
      target_size=(63, 63), class_mode='categorical', batch_size=32, shuffle=False, seed=42)


# generate the model
cnn = tf.keras.applications.EfficientNetB7(weights='imagenet', include_top=False, input_shape=(63,63,3))

for layer in cnn.layers:
  layer.trainable = True

model = tf.keras.models.Sequential([
  cnn,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(8, activation='softmax')
])

otimizador = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(loss='categorical_crossentropy', optimizer=otimizador, metrics=['acc'])

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True, verbose=1)
mc = tf.keras.callbacks.ModelCheckpoint('/content/drive/MyDrive/PSSM/CB6133/EfficientNetB7-21x21x3-63x63x3.hdf5', monitor='val_loss', save_best_only=True, verbose=1)
lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, mode='min', verbose=1)

# fit the model
model.fit(train_gen, epochs=50, batch_size=32, validation_data=val_gen, callbacks=[es, mc, lr], verbose=1)


# making predictions
# validation
pred = model.predict(val_gen, batch_size=32, verbose=1)
np.save('/content/drive/MyDrive/PSSM/CB6133/pred_efficientnetb7/21x21x3-63x63x3-val.npy', pred)

# test
pred = model.predict(test_gen, batch_size=32, verbose=1)
np.save('/content/drive/MyDrive/PSSM/CB6133/pred_efficientnetb7/21x21x3-63x63x3-test.npy', pred)