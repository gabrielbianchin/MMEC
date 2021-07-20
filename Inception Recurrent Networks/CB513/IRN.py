import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *
from tensorflow.compat.v1.keras.layers import CuDNNGRU
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam


base = np.load('/content/drive/My Drive/cullpdb+profile_6133_filtered.npy')
base = np.reshape(base, (-1, 700, 57))
a = np.arange(0,21)
b = np.arange(35,56)
c = np.hstack((a,b))

previsores = base[:, :, c]
classes = base[:, :, 22:30]

X_train = previsores[:5278,:,:]
X_val = previsores[5278:,:,:]

y_train = classes[:5278,:,:]
y_val = classes[5278:,:,:]

base = np.load('/content/drive/My Drive/cb513+profile_split1.npy')
base = np.reshape(base, (-1, 700, 57))
a = np.arange(0,21)
b = np.arange(35,56)
c = np.hstack((a,b))

previsores = base[:, :, c]
classes = base[:, :, 22:30]

X_test = previsores

y_test = classes


def conv_block(x, nb_filter, kernel_size, strides, pad):
    x = Conv1D(nb_filter, kernel_size, strides=strides, padding=pad)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def inception_A(input):

  a1 = conv_block(input, 96, 1, 1, 'same')

  a2 = conv_block(input, 64, 1, 1, 'same')
  a2 = conv_block(a2, 96, 3, 1, 'same')

  a3 = conv_block(input, 64, 1, 1, 'same')
  a3 = conv_block(a3, 96, 3, 1, 'same')
  a3 = conv_block(a3, 96, 3, 1, 'same')

  a4 = AveragePooling1D(pool_size=3, strides=1, padding='same')(input)
  a4 = conv_block(a4, 96, 1, 1, 'same')

  x = concatenate([a1, a2, a3, a4])

  return x

def inception_B(input):

  a1 = conv_block(input, 384, 1, 1, 'same')

  a2 = conv_block(input, 192, 1, 1, 'same')
  a2 = conv_block(a2, 224, 1, 1, 'same')
  a2 = conv_block(a2, 256, 7, 1, 'same')

  a3 = conv_block(input, 192, 1, 1, 'same')
  a3 = conv_block(a3, 192, 1, 1, 'same')
  a3 = conv_block(a3, 224, 7, 1, 'same')
  a3 = conv_block(a3, 224, 1, 1, 'same')
  a3 = conv_block(a3, 256, 7, 1, 'same')

  a4 = AveragePooling1D(pool_size=3, strides=1, padding='same')(input)
  a4 = conv_block(a4, 128, 1, 1, 'same')

  x = concatenate([a1, a2, a3, a4])

  return x

def inception_C(input):

  a1 = conv_block(input, 256, 1, 1, 'same')

  a2 = conv_block(input, 384, 1, 1, 'same')
  a21 = conv_block(a2, 256, 1, 1, 'same')
  a22 = conv_block(a2, 256, 3, 1, 'same')

  a3 = conv_block(input, 384, 1, 1, 'same')
  a3 = conv_block(a3, 448, 1, 1, 'same')
  a3 = conv_block(a3, 512, 3, 1, 'same')
  a31 = conv_block(a3, 256, 3, 1, 'same')
  a32 = conv_block(a3, 256, 1, 1, 'same')

  a4 = AveragePooling1D(pool_size=3, strides=1, padding='same')(input)
  a4 = conv_block(a4, 256, 1, 1, 'same')

  x = concatenate([a1, a21, a22, a31, a32, a4])

  return x

def criarRede(neurons, layers, b1=0, b2=10, b3=0):

  #aminoacid sequence
  inp1 = Input(shape=(700, 21, ))
  
  # PSSM
  inp2 = Input(shape=(700, 21, ))

  flat = Flatten()(inp1)
  emb = Embedding(14700, 1, input_length=(14700,))(flat)
  flat = Flatten()(emb)
  re = Reshape((700, 21))(flat)

  x = concatenate([re, inp2])

  for _ in range(b1):
    x = inception_A(x)
  
  for _ in range(b2):
    x = inception_B(x)

  for _ in range(b3):
    x = inception_C(x)

  gru = Bidirectional(CuDNNGRU(neurons, return_sequences=True))(x)

  for _ in range(layers-1):
    gru = Bidirectional(CuDNNGRU(neurons, return_sequences=True))(gru)
  
  out = Dense(8, activation='softmax')(gru)

  model = Model([inp1, inp2], out)

  adam = Adam(learning_rate=0.001)

  model.compile(optimizer = adam, metrics = ['acc'], loss='categorical_crossentropy')

  return model


def validate(neurons, layers, b2):
  model = criarRede(neurons, layers, b2=b2)
  es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)
  lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, mode='min', verbose=1)
  mc = ModelCheckpoint('/content/drive/My Drive/IRN/cb6133-IRN-' + str(b2) + '.hdf5', save_best_only=True)
  model.fit([X_train[:, :, :21], X_train[:, :, 21:]], y_train, epochs=50, batch_size=32, callbacks=[es, lr, mc], validation_data=([X_val[:, :, :21], X_val[:, :, 21:]], y_val))

  return model


for b2 in [3, 4, 5, 6, 7]:
  model = None
  model = validate(100, 3, b2)

  # prediction validation
  pred = model.predict([X_val[:, :, :21], X_val[:, :, 21:]])
  predicted = np.reshape(pred, (pred.shape[0] * pred.shape[1], 8)) 
  x_tst = np.reshape(X_val, (X_val.shape[0] * X_val.shape[1], X_val.shape[2]))
  
  for i in range(len(x_tst)):
    cont = 0
    for j in range(len(x_tst[i])):
      cont += x_tst[i][j]
    if cont != 0:
      y_pred.append(predicted[i])
  
  y_pred = np.array(y_pred)
  np.save('/content/drive/My Drive/IRN/cb6133-IRN-' + str(b2) +'-val.npy', y_pred)


  # prediction test
  pred = model.predict([X_test[:, :, :21], X_test[:, :, 21:]])
  predicted = np.reshape(pred, (pred.shape[0] * pred.shape[1], 8)) 
  x_tst = np.reshape(X_test, (X_test.shape[0] * X_test.shape[1], X_test.shape[2]))
  
  for i in range(len(x_tst)):
    cont = 0
    for j in range(len(x_tst[i])):
      cont += x_tst[i][j]
    if cont != 0:
      y_pred.append(predicted[i])
  
  y_pred = np.array(y_pred)
  np.save('/content/drive/My Drive/IRN/cb6133-IRN-' + str(b2) +'-test.npy', y_pred)