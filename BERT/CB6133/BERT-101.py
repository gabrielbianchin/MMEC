!pip install ktrain
import ktrain
import numpy as np
from ktrain import text

base = np.load('/content/drive/My Drive/cullpdb+profile_6133.npy')
base = np.reshape(base, (-1, 700, 57))
a = np.arange(0,21)
b = np.arange(35,56)
c = np.hstack((a,b))

previsores = base[:, :, a]
classes = base[:, :, 22:30]

X_train = previsores[:5600,:,:]
X_val = previsores[5877:6133,:,:]
X_test = previsores[5605:5877, :, :]

y_train = classes[:5600,:,:]
y_val = classes[5877:6133,:,:]
y_test = classes[5605:5877,:,:]

amino_letters = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X']
f_X_train, f_X_val, f_X_test = [], [], []
TAM = 50

for protein in X_train[:, :, :]:

  len_protein = 0
  first_amino = ''
  last_amino = ''
  for amino in protein:
    if len_protein == 0:
      first_amino = amino_letters[np.argmax(amino)]
    if amino.sum() != 0:
      len_protein += 1
      last_amino = amino_letters[np.argmax(amino)]

  for idx_amino in range(len(protein)):
    if protein[idx_amino].sum() != 0:
      start = max(0, idx_amino - TAM)
      stop = min(len_protein, idx_amino + TAM)
      s = ""

      for _ in range(TAM - idx_amino):
        s += first_amino + "+"
      for idx in range(start, stop+1):
        s += amino_letters[np.argmax(protein[idx])] + "+"
      for _ in range(idx_amino + TAM - len_protein):
        s += last_amino + "+"
      s = s.replace("+", " ")
      s = s[:-1]
      f_X_train.append(s)


for protein in X_val[:, :, :]:

  len_protein = 0
  first_amino = ''
  last_amino = ''
  for amino in protein:
    if len_protein == 0:
      first_amino = amino_letters[np.argmax(amino)]
    if amino.sum() != 0:
      len_protein += 1
      last_amino = amino_letters[np.argmax(amino)]

  for idx_amino in range(len(protein)):
    if protein[idx_amino].sum() != 0:
      start = max(0, idx_amino - TAM)
      stop = min(len_protein, idx_amino + TAM)
      s = ""

      for _ in range(TAM - idx_amino):
        s += first_amino + "+"
      for idx in range(start, stop+1):
        s += amino_letters[np.argmax(protein[idx])] + "+"
      for _ in range(idx_amino + TAM - len_protein):
        s += last_amino + "+"
      s = s.replace("+", " ")
      s = s[:-1]
      f_X_val.append(s)


for protein in X_test[:, :, :]:

  len_protein = 0
  first_amino = ''
  last_amino = ''
  for amino in protein:
    if len_protein == 0:
      first_amino = amino_letters[np.argmax(amino)]
    if amino.sum() != 0:
      len_protein += 1
      last_amino = amino_letters[np.argmax(amino)]

  for idx_amino in range(len(protein)):
    if protein[idx_amino].sum() != 0:
      start = max(0, idx_amino - TAM)
      stop = min(len_protein, idx_amino + TAM)
      s = ""

      for _ in range(TAM - idx_amino):
        s += first_amino + "+"
      for idx in range(start, stop+1):
        s += amino_letters[np.argmax(protein[idx])] + "+"
      for _ in range(idx_amino + TAM - len_protein):
        s += last_amino + "+"
      s = s.replace("+", " ")
      s = s[:-1]
      f_X_test.append(s)

f_y_train, f_y_val, f_y_test = [], [], []

for protein in y_train[:, :, :]:
  for amino in protein:
    if amino.sum() != 0:
      f_y_train.append(np.argmax(amino))

for protein in y_val[:, :, :]:
  for amino in protein:
    if amino.sum() != 0:
      f_y_val.append(np.argmax(amino))

for protein in y_test[:, :, :]:
  for amino in protein:
    if amino.sum() != 0:
      f_y_test.append(np.argmax(amino))

MODEL_NAME = 'Rostlab/prot_bert_bfd'
t = text.Transformer(MODEL_NAME, maxlen=101, classes=[0, 1, 2, 3, 4, 5, 6, 7])
trn = t.preprocess_train(f_X_val, f_y_val)
val = t.preprocess_test(f_X_val, f_y_val)
model = t.get_classifier()

learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=32)
learner.autofit(1e-5, 5, early_stopping=1, checkpoint_folder='/content/drive/MyDrive/prot_bert_bfd_cb6133/101')
predictor = ktrain.get_predictor(learner.model, preproc=t)

predict = predictor.predict_proba(f_X_val)
predict = np.array(predict)
np.save('/content/drive/MyDrive/prot_bert_bfd_cb6133/bertprot-cb6133-window-101-val.npy', predict)

predict = predictor.predict_proba(f_X_test)
predict = np.array(predict)
np.save('/content/drive/MyDrive/prot_bert_bfd_cb6133/bertprot-cb6133-window-101-test.npy', predict)