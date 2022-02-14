import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split

plt.style.use('ggplot')
stop_words = set(stopwords.words('english'))

from tqdm.notebook import tqdm

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv')
train = train.drop(columns = ['id', 'keyword', 'location'], axis = 1)

def clear_text(text):
    text = word_tokenize(text)
    text = [word for word in text if word.isalpha()]
    text = [word.lower() for word in text]
    text = [word for word in text if word not in stop_words]
    result = ''
    for word in text:
        result += word + ' '
    return(result)

train_target, test_target, train_text, test_text = train_test_split(train['target'],
                                                                   train['text'],
                                                                   shuffle = False,
                                                                   test_size = 0.2)
test_target, val_target, test_text, val_text = train_test_split(test_target,
                                                                test_text,
                                                                shuffle = False)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 10000
max_len = 120
embedding_dim = 16
trunk_type = 'post'
oov_tok = '<OOV>'

tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)
tokenizer.fit_on_texts(train_text)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(train_text)
padded = pad_sequences(sequences, maxlen = max_len, truncating = trunk_type)

val_sequences = tokenizer.texts_to_sequences(val_text)
val_padded = pad_sequences(val_sequences, maxlen = max_len, truncating = trunk_type)

test_sequences = tokenizer.texts_to_sequences(test_text)
test_padded = pad_sequences(test_sequences, maxlen = max_len, truncating = trunk_type)
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_len),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(14, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()

num_epochs = 15
history = model.fit(padded, train_target, epochs = num_epochs, validation_data = (val_padded, val_target))

model.save_weights('models/tensorflow.h5')
def plot(history,string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()
plot(history, "accuracy")
plot(history, 'loss')

model.load_weights('models/tensorflow.h5')
preds = model(test_padded).numpy()

results = pd.DataFrame(columns = ('predictions', 'true'))
results['predictions'] = preds.flatten()
results['predictions'] = round(results['predictions'])
results['true'] = np.array(test_target)
correct = (results['predictions'] == results['true']).sum()
print(f'accuracy: {100 * correct/len(preds)}')
results.to_csv('results/tensorflow_model_results')