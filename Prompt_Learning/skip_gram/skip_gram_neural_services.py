import numpy as np
from keras.layers import *
from keras.layers.core import Dense, Reshape
from keras.layers import Embedding
from keras.models import Model,Sequential
import pandas as pd

def generate_dataset(service_df, word2id, V):
    services_1 = []
    services_2 = []

    services = unique(list(service_df['service_1']))

    start = 0
    stop = 0

    for i in range(len(services)):
        service_1 = services[i]
        stop += (V - 1)
        for j in range(start, stop):
            service_2 = service_df['service_2'][j]
            services_1.append(word2id[service_1])
            services_2.append(word2id[service_2])

        start += (V - 1)

    return services_1, services_2

###############################################################################
# Remove duplicate
###############################################################################

def unique(list1):
    x = np.array(list1)
    return list(np.unique(x))

col_names = ['service_1', 'service_2', 'target']
service_df = pd.read_csv('training_set_skip_gram_85.csv', skiprows=1, sep=';', names=col_names, encoding ="ISO-8859-1")

services = unique(list(service_df['service_1']))
V = len(services) + 1
word2id = {}
for i in range(len(services)):
    word2id[services[i]] = i

print('Vocabulary Size:', V)
print('Vocabulary Sample:', list(word2id.items())[:5])
id2word = {v:k for k, v in word2id.items()}

pairs = generate_dataset(service_df, word2id, V)

###############################################################################
# Build skip-gram architecture
###############################################################################

embed_size = 768 # GPT-2 embedding size
word_model = Sequential()
word_model.add(Embedding(V, embed_size,
                      embeddings_initializer="glorot_uniform",
                      input_length=1))
word_model.add(Reshape((embed_size, )))
context_model = Sequential()
context_model.add(Embedding(V, embed_size,
               embeddings_initializer="glorot_uniform",
               input_length=1))
context_model.add(Reshape((embed_size,)))
merged_output = add([word_model.output, context_model.output])
model_combined = Sequential()
model_combined.add(Dense(1, kernel_initializer="glorot_uniform", activation="sigmoid"))
final_model = Model([word_model.input, context_model.input], model_combined(merged_output))
final_model.compile(loss="binary_crossentropy", optimizer="rmsprop")
final_model.summary()

final_model.fit(x=[np.array(pairs[0], dtype='int32'), np.array(pairs[1], dtype='int32')],
                y=np.array(service_df['target'], dtype='int32'), batch_size=80, epochs=50, verbose=1)

word_embed_layer = word_model.layers[0]
weights = word_embed_layer.get_weights()[0][1:]

###############################################################################
# Save service embeddings
###############################################################################

weights_df = pd.DataFrame(weights, index=word2id)
print(weights_df.head())

weights_df.to_csv('./service_weights_85.csv')