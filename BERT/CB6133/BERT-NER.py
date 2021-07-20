import numpy as np
!pip -q install transformers seqeval
from transformers import Trainer, TrainingArguments, AutoModelForTokenClassification, BertTokenizerFast, EvalPrediction
from torch.utils.data import Dataset
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.nn.functional as F 

# loading database
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

# transforming numbers to letters (aminoacid and ss)
amino_letters = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X']
f_X_train, f_X_val, f_X_test = [], [], []

for protein in X_train[:, :, :]:
  p = []
  for idx_amino in range(len(protein)):
    if protein[idx_amino].sum() != 0:
      p.append(amino_letters[np.argmax(protein[idx_amino])])
  f_X_train.append(p)

for protein in X_val[:, :, :]:
  p = []
  for idx_amino in range(len(protein)):
    if protein[idx_amino].sum() != 0:
      p.append(amino_letters[np.argmax(protein[idx_amino])])
  f_X_val.append(p)

for protein in X_test[:, :, :]:
  p = []
  for idx_amino in range(len(protein)):
    if protein[idx_amino].sum() != 0:
      p.append(amino_letters[np.argmax(protein[idx_amino])])
  f_X_test.append(p)


SS_letters = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T']
f_y_train, f_y_val, f_y_test = [], [], []

for ss in y_train[:, :, :]:
  p = []
  for idx_ss in range(len(ss)):
    if ss[idx_ss].sum() != 0:
      p.append(SS_letters[np.argmax(ss[idx_ss])])
  f_y_train.append(p)

for ss in y_val[:, :, :]:
  p = []
  for idx_ss in range(len(ss)):
    if ss[idx_ss].sum() != 0:
      p.append(SS_letters[np.argmax(ss[idx_ss])])
  f_y_val.append(p)

for ss in y_test[:, :, :]:
  p = []
  for idx_ss in range(len(ss)):
    if ss[idx_ss].sum() != 0:
      p.append(SS_letters[np.argmax(ss[idx_ss])])
  f_y_test.append(p)



# preprocessing steps BERT
seq_tokenizer = BertTokenizerFast.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)

train_seqs_encodings = seq_tokenizer(f_X_train, is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=True)
val_seqs_encodings = seq_tokenizer(f_X_val, is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=True)
test_seqs_encodings = seq_tokenizer(f_X_test, is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=True)

id2tag = {0: 'L', 1: 'B', 2: 'E', 3: 'G', 4:'I', 5:'H', 6:'S', 7:'T'}
tag2id = {id: tag for tag, id in id2tag.items()}
unique_tags = set(tag for doc in f_y_train for tag in doc)
tag2id, id2tag, unique_tags

def encode_tags(tags, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels

train_labels_encodings = encode_tags(f_y_train, train_seqs_encodings)
val_labels_encodings = encode_tags(f_y_val, val_seqs_encodings)
test_labels_encodings = encode_tags(f_y_test, test_seqs_encodings)


class SSDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

_ = train_seqs_encodings.pop("offset_mapping")
_ = val_seqs_encodings.pop("offset_mapping")
_ = test_seqs_encodings.pop("offset_mapping")
train_dataset = SSDataset(train_seqs_encodings, train_labels_encodings)
val_dataset = SSDataset(val_seqs_encodings, val_labels_encodings)
test_dataset = SSDataset(test_seqs_encodings, test_labels_encodings)


def align_predictions(predictions: np.ndarray, label_ids: np.ndarray):
        preds = np.argmax(predictions, axis=2)

        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != torch.nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(id2tag[label_ids[i][j]])
                    preds_list[i].append(id2tag[preds[i][j]])

        return preds_list, out_label_list

def compute_metrics(p: EvalPrediction):
    preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)
    return {
        "accuracy": accuracy_score(out_label_list, preds_list),
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }


def model_init():
  return AutoModelForTokenClassification.from_pretrained('Rostlab/prot_bert_bfd',
                                                         num_labels=len(unique_tags),
                                                         id2label=id2tag,
                                                         label2id=tag2id,
                                                         gradient_checkpointing=False)

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=5,              # total number of training epochs
    per_device_train_batch_size=1,   # batch size per device during training
    per_device_eval_batch_size=8,   # batch size for evaluation
    warmup_steps=200,                # number of warmup steps for learning rate scheduler
    learning_rate=1e-05,             # learning rate
    weight_decay=0.0,                # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,               # How often to print logs
    do_train=True,                   # Perform training
    do_eval=True,                    # Perform evaluation
    evaluation_strategy="epoch",     # evalute after each epoch
    run_name="ProBert-BFD-SS3",      # experiment name
    seed=3,                         # Seed for experiment reproducibility
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,

)

trainer = Trainer(
    model_init=model_init,                # the instantiated 🤗 Transformers model to be trained
    args=training_args,                   # training arguments, defined above
    train_dataset=train_dataset,          # training dataset
    eval_dataset=val_dataset,             # evaluation dataset
    compute_metrics = compute_metrics,    # evaluation metrics
)

# train
trainer.train()


# prediction on validation
predictions, label_ids, metrics = trainer.predict(val_dataset)

pre = []
for prot in range(label_ids.shape[0]):
  for amino in range(label_ids.shape[1]):
    if label_ids[prot][amino] != -100:
      pre.append(predictions[prot][amino])

pred = F.softmax(torch.tensor(pre))
np.save('/content/drive/MyDrive/prot_bert_bfd_ner_cb6133/pred-val.npy', pred)


# prediction on test
predictions, label_ids, metrics = trainer.predict(test_dataset)

pre = []
for prot in range(label_ids.shape[0]):
  for amino in range(label_ids.shape[1]):
    if label_ids[prot][amino] != -100:
      pre.append(predictions[prot][amino])

pred = F.softmax(torch.tensor(pre))
np.save('/content/drive/MyDrive/prot_bert_bfd_ner_cb6133/pred-test.npy', pred)