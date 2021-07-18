import pandas as pd

df_val = pd.read_csv('val_summ_raw.csv')
df_test = pd.read_csv('test_summ_raw.csv')

df = pd.read_csv('train_summ_raw.csv')
df.head()

from sklearn.feature_extraction.text import TfidfVectorizer


tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2',max_features=512, encoding='latin-1', ngram_range=(1, 2), stop_words='english')

features = tfidf.fit_transform(df.tweet).toarray()

d = {'real': 1, 'fake': 0}

labels = [d[i] for i in df.label]
val_labels = [d[i] for i in df_val.label]
test_labels = [d[i] for i in df_test.label]

x_val = tfidf.transform(df_val.tweet)
x_test = tfidf.transform(df_test.tweet)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

lr = RandomForestClassifier(max_depth=50, n_estimators=100)

lr.fit(features, labels)

svc = SVC()
svc.fit(features, labels)

lr = LogisticRegression(random_state=0)

lr.fit(features, labels)


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score, recall_score, f1_score


print('Acc: ', accuracy_score(test_labels, pred))
print('Precision: ', precision_score(test_labels, pred, average='weighted'))
print('Recall:', recall_score(test_labels, pred,  average='weighted'))
print('F1: ', f1_score(test_labels, pred,  average='weighted'))

print('Acc: ', accuracy_score(test_labels, lr.predict(x_test)))
print('Precision: ', precision_score(test_labels, lr.predict(x_test), average='weighted'))
print('Recall:', recall_score(test_labels, lr.predict(x_test),  average='weighted'))
print('F1: ', f1_score(test_labels, lr.predict(x_test),  average='weighted'))

from keras.datasets import imdb
import pandas as pd
import numpy as np
from keras.layers import LSTM, Activation, Dropout, Dense, Input, Bidirectional, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.models import Model
import string
import re
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.sequence import pad_sequences
import keras
from sklearn.model_selection import train_test_split

data = pd.read_csv('train_summ_raw.csv')
val_data = pd.read_csv('val_summ_raw.csv')
test_data = pd.read_csv('test_summ_raw.csv')

data['tweet'] = data['tweet'].str.lower()
val_data['tweet'] = val_data['tweet'].str.lower()
test_data['tweet'] = test_data['tweet'].str.lower()


stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because",
             "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during",
             "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here",
             "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into",
             "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or",
             "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should",
             "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's",
             "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up",
             "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's",
             "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've",
             "your", "yours", "yourself", "yourselves" ]


def remove_stopwords(data):
    data['review without stopwords'] = data['tweet'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
    return data


def remove_tags(string):
    result = re.sub('<.*?>', '', string)
    return result


data_without_stopwords = remove_stopwords(data)
data_without_stopwords['clean_review'] = data_without_stopwords['review without stopwords'].apply(
    lambda cw: remove_tags(cw))
# data_without_stopwords['clean_review'] = data_without_stopwords['clean_review'].str.replace('{}'.format(string.punctuation), ' ')

val_data_without_stopwords = remove_stopwords(val_data)
val_data_without_stopwords['clean_review'] = val_data_without_stopwords['review without stopwords'].apply(
    lambda cw: remove_tags(cw))

test_data_without_stopwords = remove_stopwords(test_data)
test_data_without_stopwords['clean_review'] = test_data_without_stopwords['review without stopwords'].apply(
    lambda cw: remove_tags(cw))


def remove_stopwords(data):
    data['review without stopwords'] = data['tweet'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
    return data


def remove_tags(string):
    result = re.sub('<.*?>', '', string)
    return result


data_without_stopwords = remove_stopwords(data)
data_without_stopwords['clean_review'] = data_without_stopwords['review without stopwords'].apply(
    lambda cw: remove_tags(cw))
# data_without_stopwords['clean_review'] = data_without_stopwords['clean_review'].str.replace('{}'.format(string.punctuation), ' ')

val_data_without_stopwords = remove_stopwords(val_data)
val_data_without_stopwords['clean_review'] = val_data_without_stopwords['review without stopwords'].apply(
    lambda cw: remove_tags(cw))

test_data_without_stopwords = remove_stopwords(test_data)
test_data_without_stopwords['clean_review'] = test_data_without_stopwords['review without stopwords'].apply(
    lambda cw: remove_tags(cw))
y_train = np.array(list(map(lambda x: 1 if x=="real" else 0, data_without_stopwords.label)))
y_val = np.array(list(map(lambda x: 1 if x=="real" else 0, val_data_without_stopwords.label)))
y_test = np.array(list(map(lambda x: 1 if x=="real" else 0, test_data_without_stopwords.label)))

X_train = data_without_stopwords['clean_review']
X_val = val_data_without_stopwords['clean_review']
X_test = test_data_without_stopwords['clean_review']

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

words_to_index = tokenizer.word_index

def read_glove_vector(glove_vec):
    with open(glove_vec, 'r', encoding='UTF-8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            w_line = line.split()
            curr_word = w_line[0]
            word_to_vec_map[curr_word] = np.array(w_line[1:], dtype=np.float64)

    return word_to_vec_map

word_to_vec_map = read_glove_vector('glove.6B.300d.txt')
maxLen = 512


vocab_len = len(words_to_index)
embed_vector_len = word_to_vec_map['moon'].shape[0]
print(embed_vector_len)
emb_matrix = np.zeros((vocab_len+1, embed_vector_len))

for word, index in words_to_index.items():
#     print(index)
    embedding_vector = word_to_vec_map.get(word)
#     print(embedding_vector)
    if embedding_vector is not None:
        emb_matrix[index, :] = embedding_vector




embedding_layer = Embedding(input_dim=vocab_len+1, output_dim=embed_vector_len, input_length=maxLen, weights = [emb_matrix], trainable=False)



X_indices = Input(shape=(512, ))

embeddings = embedding_layer(X_indices)
# embeddings = Embedding(input_dim=vocab_len+1, output_dim=embed_vector_len, input_length=maxLen)(X_indices)


X = Bidirectional(LSTM(128, return_sequences=True))(embeddings)

X = Dropout(0.2)(X)
# X = Dense(64)(X)

X = keras.layers.Flatten()(X)


# X = LSTM(128, return_sequences=True)(X)

# X = Dropout(0.6)(X)

# X = LSTM(128)(X)

X = Dense(1, activation='sigmoid')(X)

model = Model(inputs=X_indices, outputs=X)

X_train_indices = tokenizer.texts_to_sequences(X_train)


X_train_indices = pad_sequences(X_train_indices, maxlen=maxLen, padding='post')



X_val_indices = tokenizer.texts_to_sequences(X_val)
X_val_indices = pad_sequences(X_val_indices, maxlen=maxLen, padding='post')



adam = keras.optimizers.Adam(lr = 0.001)


model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train_indices, y_train, batch_size=256, epochs=8, validation_data=(X_val_indices, y_val))

X_test_indices = tokenizer.texts_to_sequences(X_test)
X_test_indices = pad_sequences(X_test_indices, maxlen=maxLen, padding='post')

pred = [1 if i>0.5 else 0 for i in model.predict(X_test_indices)]

print('Acc: ', accuracy_score(y_test, pred))
print('Precision: ', precision_score(y_test, pred, average='weighted'))
print('Recall:', recall_score(y_test, pred,  average='weighted'))
print('F1: ', f1_score(y_test, pred,  average='weighted'))