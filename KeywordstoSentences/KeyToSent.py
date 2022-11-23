import ftfy
from keytotext import trainer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_string(keywords):
  string = ""
  for kw in keywords:
    if len(kw) > 1:
      string = string + " " + kw.replace(" ", "_")
    else:
      string = string + " " + kw

  return string[1:]

def getTarget(id):
   if id == 1:
    return "Personal"
   elif id == 2:
    return "Physical"
   else:
    return "Cybersecurity"

###############################################################################
# Extract candidates
###############################################################################

def keyWordOperation(stringa, n_keywords):
  count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([stringa])
  all_candidates = count.get_feature_names()

  candidate_tokens = tokenizer(all_candidates, padding=True, return_tensors="pt")
  candidate_embeddings = model(**candidate_tokens)["pooler_output"]

  text_tokens = tokenizer([stringa], padding=True, return_tensors="pt")
  text_embedding = model(**text_tokens)["pooler_output"]

  candidate_embeddings = candidate_embeddings.detach().numpy()
  text_embedding = text_embedding.detach().numpy()

  distances = cosine_similarity(text_embedding, candidate_embeddings)
  keywords = [all_candidates[index] for index in distances.argsort()[0][-n_keywords:]]

  listaDiKeyChannel = extract_string(keywords)

  return listaDiKeyChannel

###############################################################################
# Load data
###############################################################################

col_names = ['triggerTitle','triggerChannelTitle','actionTitle','actionChannelTitle','title','desc','target','motivation']

df_path = './full_dataset.csv'
df = pd.read_csv(df_path,skiprows=1,sep=';',names=col_names,encoding = "ISO-8859-1")

train_path = './training_dataset.csv'
training_set = pd.read_csv(train_path,skiprows=1,sep=';',names=col_names,encoding = "ISO-8859-1")

test_path = './test_dataset.csv'
test_set = pd.read_csv(test_path,skiprows=1,sep=';',names=col_names,encoding = "ISO-8859-1")

val_path = './val_dataset.csv'
val_set = pd.read_csv(val_path,skiprows=1,sep=';',names=col_names,encoding = "ISO-8859-1")

###############################################################################
# Load DistilRoBERTa
###############################################################################

model_name = "distilroberta-base"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
n_gram_range = (1, 2)
stop_words = "english"
n_keywords = 4

###############################################################################
# Extract candidates from training data
###############################################################################

keywords_training = []
i = 0
while i < len(training_set):
    channel = "if " + ftfy.fix_text(training_set.iloc[i, 0]) + "(" + ftfy.fix_text(
        training_set.iloc[i, 1]) + ") then " + ftfy.fix_text(training_set.iloc[i, 2]) + "(" + ftfy.fix_text(
        training_set.iloc[i, 3]) + ")"
    print(channel)
    title = ftfy.fix_text(training_set.iloc[i, 4])
    print(title)
    desc = ftfy.fix_text(training_set.iloc[i, 5])
    print(desc)

    try:
        listKeyChannel = keyWordOperation(channel, n_keywords)
    except:
        listKeyChannel = ""

    try:
        listKeyTitle = keyWordOperation(title, n_keywords)
    except:
        listKeyTitle = ""

    try:
        listKeyDesc = keyWordOperation(desc, n_keywords)
    except:
        listKeyDesc = ""

    keywords = "" + listKeyChannel + " " + listKeyTitle + " " + listKeyDesc + " " + getTarget(training_set.iloc[i, 6])

    listTotSplit = keywords.split()
    listTot = " ".join(sorted(set(listTotSplit), key=listTotSplit.index))

    print(listTot)

    keywords_training.append(listTot)
    print(i)
    i += 1

train_df = pd.DataFrame({'keywords':keywords_training, 'rule':training_set['desc'], 'text':training_set['motivation']})

###############################################################################
# Extract candidates from test data
###############################################################################

keywords_test = []
i = 0
while i < len(test_set):
    channel = "if " + ftfy.fix_text(test_set.iloc[i, 0]) + "(" + ftfy.fix_text(
        test_set.iloc[i, 1]) + ") then " + ftfy.fix_text(test_set.iloc[i, 2]) + "(" + ftfy.fix_text(
        test_set.iloc[i, 3]) + ")"
    print(channel)
    title = ftfy.fix_text(test_set.iloc[i, 4])
    print(title)
    desc = ftfy.fix_text(test_set.iloc[i, 5])
    print(desc)

    try:
        listKeyChannel = keyWordOperation(channel, n_keywords)
    except:
        listKeyChannel = ""

    try:
        listKeyTitle = keyWordOperation(title, n_keywords)
    except:
        listKeyTitle = ""

    try:
        listKeyDesc = keyWordOperation(desc, n_keywords)
    except:
        listKeyDesc = ""

    keywords = "" + listKeyChannel + " " + listKeyTitle + " " + listKeyDesc + " " + getTarget(
        test_set.iloc[i, 6])

    listTotSplit = keywords.split()
    listaTot = " ".join(sorted(set(listTotSplit), key=listTotSplit.index))

    print(listaTot)

    keywords_test.append(listaTot)
    print(i)
    i += 1

test_df = pd.DataFrame({'keywords':keywords_test, 'rule':test_set['desc'], 'text':test_set['motivation']})

###############################################################################
# Extract candidates from validation data
###############################################################################

keywords_val = []
i = 0
while i < len(val_set):
    channel = "if " + ftfy.fix_text(val_set.iloc[i, 0]) + "(" + ftfy.fix_text(
        val_set.iloc[i, 1]) + ") then " + ftfy.fix_text(val_set.iloc[i, 2]) + "(" + ftfy.fix_text(
        val_set.iloc[i, 3]) + ")"
    print(channel)
    title = ftfy.fix_text(val_set.iloc[i, 4])
    print(title)
    desc = ftfy.fix_text(val_set.iloc[i, 5])
    print(desc)

    try:
        listKeyChannel = keyWordOperation(channel, n_keywords)
    except:
        listKeyChannel = ""

    try:
        listKeyTitle = keyWordOperation(title, n_keywords)
    except:
        listKeyTitle = ""

    try:
        listKeyDesc = keyWordOperation(desc, n_keywords)
    except:
        listKeyDesc = ""

    keywords = "" + listKeyChannel + " " + listKeyTitle + " " + listKeyDesc + " " + getTarget(
        val_set.iloc[i, 6])

    listTotSplit = keywords.split()
    listTot = " ".join(sorted(set(listTotSplit), key=listTotSplit.index))

    print(listaTot)

    keywords_val.append(listaTot)
    print(i)
    i += 1

val_df = pd.DataFrame({'keywords':keywords_val, 'rule':val_set['desc'], 'text':val_set['motivation']})

###############################################################################
# Training code
###############################################################################

model = trainer()
model.from_pretrained(model_name="t5-small")
model.train(train_df=train_df, test_df=val_df, batch_size=3, max_epochs=3,use_gpu=True)

###############################################################################
# Save model
###############################################################################

model.save_model('./ModelKeyToText_evaluated_4')
model = trainer()

###############################################################################
# Load model
###############################################################################

model.load_model("./ModelKeyToText_evaluated_4", use_gpu=True)

###############################################################################
# Run on test data
###############################################################################

gold_motivations = test_df['text']
generated = []

for i in range(0, len(test_df)):
    keywords=[test_df['keywords'][i]]
    print("Description: ", test_df['rule'][i])
    print("Keywords extracted: ", keywords)
    new_motivation = model.predict(keywords)
    generated.append(new_motivation)
    print("Gold motivation: ",test_df['text'][i])
    print("New motivation: ", new_motivation)

test_set['motivation_generated'] = generated
test_set.to_csv('test_set_results_T5_KeyToText_evaluated_4.csv')