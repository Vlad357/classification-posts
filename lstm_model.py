import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

from sklearn.model_selection import train_test_split

plt.style.use('ggplot')
stop_words = set(stopwords.words('english'))

from tqdm.notebook import tqdm

import warnings
warnings.filterwarnings('ignore')

data_df = pd.read_csv('train.csv')

data_df = data_df.drop(columns = ['id', 'keyword', 'location'], axis = 1)


def clear_text(text):
    text = word_tokenize(text)
    text = [word.lower() for word in text]
    text = [word.replace(r"http\S+", "URL") for word in text]
    text = [word.replace(r'[^A-Za-z0-9()!?\'\`\"]', 'and') for word in text]
    text = [word.replace('\s{2,}', 'and') for word in text]
    text = [word.replace(r'\#', 'and') for word in text]
    text = [word.replace(r'@', 'and') for word in text]
    text = [word for word in text if not word in stop_words]
    concat = ''
    for word in text:
        concat += word + ' '

    return concat

data_df['text'] = data_df['text'].apply(lambda x: clear_text(x))

train_data, test_data = train_test_split(data_df, shuffle = False)
val_data, test_data = train_test_split(test_data, shuffle = False, train_size = 1000)

import torch
from torch import nn
from torchtext.legacy import data

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

TEXT = data.Field(sequential = True, include_lengths = True)
LABEL = data.LabelField(dtype = torch.float)


class DataFrameDataset(data.Dataset):
    def __init__(self, df, fields, is_test=False, **kwargs):
        examples = []
        for i, row in df.iterrows():
            label = row.target if not is_test else None
            text = row.text
            examples.append(data.Example.fromlist([text, label], fields))

        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, fields, train_dataFrame, val_dataFrame, test_dataFrame=None, **kwargs):
        train_data, val_data, test_data = (None, None, None)
        data_field = fields
        if train_dataFrame is not None:
            train_data = cls(train_dataFrame.copy(), data_field, **kwargs)
        if val_dataFrame is not None:
            val_data = cls(val_dataFrame.copy(), data_field, **kwargs)
        if test_dataFrame is not None:
            test_data = cls(test_dataFrame.copy(), data_field, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data) if d is not None)

fields = [('text', TEXT), ('label', LABEL)]
train_ds, val_ds, test_ds = DataFrameDataset.splits(fields,
                                                    train_dataFrame = train_data,
                                                    val_dataFrame = val_data,
                                                    test_dataFrame = test_data,)

MAX_VOCAB_SIZE = 25000

TEXT.build_vocab(train_ds,
                max_size = MAX_VOCAB_SIZE,
                vectors = 'glove.6B.200d',
                unk_init = torch.Tensor.zero_)
LABEL.build_vocab(train_ds)
batch_size = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, val_iterator, test_iterator = data.BucketIterator.splits((train_ds, val_ds, test_ds),
                                                          batch_size = batch_size,
                                                          sort_within_batch = True,
                                                          device = device)
num_epochs = 30
learning_rate = 0.001

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 200
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.2
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]


class LSTM_model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim * n_layers, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        pasked_output, (hidden, cell) = self.rnn(packed_embedded)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        output = self.fc1(hidden)
        output = self.dropout(self.fc2(output))
        return output
model = LSTM_model(INPUT_DIM,
                  EMBEDDING_DIM,
                  HIDDEN_DIM,
                  OUTPUT_DIM,
                  N_LAYERS,
                  BIDIRECTIONAL,
                  DROPOUT,
                  PAD_IDX)
pretrained_embeddings = TEXT.vocab.vectors

model.embedding.weight.data.copy_(pretrained_embeddings)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
model.to(device)
criterion = nn.BCEWithLogitsLoss()
optim = torch.optim.Adam(model.parameters(), lr = learning_rate)


def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

def train(model, iterator):
    ep_loss = 0
    ep_acc = 0

    model.train()

    for batch in iterator:
        text, text_lengths = batch.text

        optim.zero_grad()
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)

        loss.backward()
        optim.step()

        ep_loss += loss.item()
        ep_acc += acc.item()

    return ep_loss / len(iterator), ep_acc / len(iterator)


def evalute(model, iterator):
    ep_acc = 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            preds = model(text, text_lengths).squeeze(1)
            acc = binary_accuracy(preds, batch.label)

            ep_acc += acc.item()

    return ep_acc / len(iterator)


loss = []
acc = []
val_acc = []
for epoch in range(num_epochs):
    print(f'epoch {epoch + 1}/{num_epochs}')

    train_loss, train_acc = train(model, train_iterator)
    valid_acc = evalute(model, val_iterator)

    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Acc: {valid_acc * 100:.2f}%')

    loss.append(train_loss)
    acc.append(train_acc)
    val_acc.append(valid_acc)
torch.save(model, 'models/torch_lstmModel_v1')

plt.title('learning results')
plt.xlabel('epochs')
plt.ylabel('loss, validation accuracy, accuracy')
plt.plot(range(num_epochs), loss, color='red', label='loss')
plt.plot(range(num_epochs), val_acc, color='green', label='validation accuracy')
plt.plot(range(num_epochs), acc, color='yellow', label='accuracy')
plt.legend()
plt.show()

model = torch.load('models/torch_lstmModel_v1')

prediction = []
true = []
for batch in test_iterator:
    text, text_lengths = batch.text
    prediction.extend(list(model(text, text_lengths).detach().numpy().flatten()))
    true.extend(list((batch.label).detach().numpy().flatten()))

results = pd.DataFrame(columns=('predictions', 'true'))
results['predictions'] = prediction
results['predictions'] = round(results['predictions'])
results['true'] = true
# accuracy
for pred in results['predictions']:
    pred = 0 if pred < 0.5 else 1
correct = (results['predictions'] == results['true']).sum()
print(f'accuracy: {100 * correct / len(true)}')

results.to_csv('results/results_lstm_v1_torch')