import pandas as pd
import ftfy
import re
from sentence_transformers import SentenceTransformer
import numpy as np
import io, csv

###############################################################################
# Cosine similarity
###############################################################################

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

###############################################################################
# Description cleaning function
###############################################################################

def clear_descriptions(services_description):
    cleaned_services_description = []

    emoj = re.compile("["
                      u"\U0001F600-\U0001F64F"  # emoticons
                      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                      u"\U0001F680-\U0001F6FF"  # transport & map symbols
                      u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                      u"\U00002500-\U00002BEF"  # chinese char
                      u"\U00002702-\U000027B0"
                      u"\U00002702-\U000027B0"
                      u"\U000024C2-\U0001F251"
                      u"\U0001f926-\U0001f937"
                      u"\U00010000-\U0010ffff"
                      u"\u2640-\u2642"
                      u"\u2600-\u2B55"
                      u"\u200d"
                      u"\u23cf"
                      u"\u23e9"
                      u"\u231a"
                      u"\ufe0f"  # dingbats
                      u"\u3030"
                      "]+", re.UNICODE)

    for description in services_description:
        temp_description = ftfy.fix_text(description)
        temp_description = re.sub(emoj, '', temp_description)
        temp_description = temp_description.replace('\n', "")
        cleaned_services_description.append(temp_description.replace('\t', ""))

    return cleaned_services_description

###############################################################################
# Load data
###############################################################################

col_names = ['service','description']
service_df = pd.read_csv('./services_description.csv',skiprows=1,sep=';',names=col_names,encoding = "ISO-8859-1")

cleaned_services_description = clear_descriptions(service_df['description'])

###############################################################################
# Load SentenceBERT
###############################################################################

sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
sentence_embeddings = sbert_model.encode(cleaned_services_description)

threshold = 0.85

services_1 = []
services_2 = []
target = []

###############################################################################
# Definition of Skip-Gram training pairs (target service, context service)
###############################################################################

for service_main in range(len(service_df['service'])):
    for service_target in range(len(service_df['service'])):
        services_1.append(service_df['service'][service_main])
        services_2.append(service_df['service'][service_target])
        if cosine(sentence_embeddings[service_main], sentence_embeddings[service_target]) >= threshold:
            target.append(1)
            print(service_df['service'][service_main], "and", service_df['service'][service_target], "are similar")
        else:
            target.append(0)
            print(service_df['service'][service_main], "and", service_df['service'][service_target], "are not similar")

col_names = ['service_1', 'service_2', 'target']

with io.open('./training_set_skip_gram_85.csv', mode='w', newline='', encoding='utf-8') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=col_names, delimiter=";")
    writer.writeheader()
    for i in range(len(target)):
        writer = csv.DictWriter(csv_file, fieldnames=col_names, delimiter=";")
        writer.writerow({'service_1': services_1[i], 'service_2': services_2[i], 'target': target[i]})

