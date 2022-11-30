import ftfy, os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration,Adafactor

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

def generate(text,model,tokenizer):
   model.eval()
   input_ids = tokenizer.encode("justification generation: {}</s>".format(text),
                               return_tensors="pt").to(dev)
   outputs = model.generate(input_ids)

   return tokenizer.decode(outputs[0])

###############################################################################
# Extract candidates
###############################################################################

def keyWordOperation(string, n_keywords, n_gram_range, stop_words):
  count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([string])
  all_candidates = count.get_feature_names()

  candidate_tokens = tokenizer(all_candidates, padding=True, return_tensors="pt")
  candidate_embeddings = model(**candidate_tokens)["pooler_output"]

  text_tokens = tokenizer([string], padding=True, return_tensors="pt")
  text_embedding = model(**text_tokens)["pooler_output"]

  candidate_embeddings = candidate_embeddings.detach().numpy()
  text_embedding = text_embedding.detach().numpy()

  distances = cosine_similarity(text_embedding, candidate_embeddings)
  keywords = [all_candidates[index] for index in distances.argsort()[0][-n_keywords:]]

  return extract_string(keywords)

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
n_keywords = 4 # 5, 6, 7, 8

###############################################################################
# Extract candidates from training data
###############################################################################

keywords_training = []
i = 0
while i < len(training_set):
    channel = "if " + ftfy.fix_text(training_set.iloc[i, 0]) + " then " + ftfy.fix_text(training_set.iloc[i, 2])
    title = ftfy.fix_text(training_set.iloc[i, 4])
    desc = ftfy.fix_text(training_set.iloc[i, 5])

    try:
        listKeyChannel = keyWordOperation(channel, n_keywords, n_gram_range, stop_words) + " " + ftfy.fix_text(
        training_set.iloc[i, 1]) + " " + ftfy.fix_text(training_set.iloc[i, 3])
    except:
        listKeyChannel = ""

    try:
        listKeyTitle = keyWordOperation(title, n_keywords, n_gram_range, stop_words)
    except:
        listKeyTitle = ""

    try:
        listKeyDesc = keyWordOperation(desc, n_keywords, n_gram_range, stop_words)
    except:
        listKeyDesc = ""

    keywords = "" + listKeyChannel + " " + listKeyTitle + " " + listKeyDesc + " " + getTarget(training_set.iloc[i, 6])

    listTotSplit = keywords.split()
    listTot = " | ".join(sorted(set(listTotSplit), key=listTotSplit.index))

    print("ListTot:", listTot)

    keywords_training.append(listTot)
    i += 1

train_df = pd.DataFrame({'keywords':keywords_training, 'rule':val_set['desc'], 'text':training_set['motivation']})

###############################################################################
# Extract candidates from test data
###############################################################################

keywords_test = []
i = 0
while i < len(test_set):
    channel = "if " + ftfy.fix_text(test_set.iloc[i, 0]) + " then " + ftfy.fix_text(test_set.iloc[i, 2])
    title = ftfy.fix_text(test_set.iloc[i, 4])
    desc = ftfy.fix_text(test_set.iloc[i, 5])

    try:
        listKeyChannel = keyWordOperation(channel, n_keywords, n_gram_range, stop_words) + " " + ftfy.fix_text(
            training_set.iloc[i, 1]) + " " + ftfy.fix_text(training_set.iloc[i, 3])
    except:
        listKeyChannel = ""

    try:
        listKeyTitle = keyWordOperation(title, n_keywords, n_gram_range, stop_words)
    except:
        listKeyTitle = ""

    try:
        listKeyDesc = keyWordOperation(desc, n_keywords, n_gram_range, stop_words)
    except:
        listKeyDesc = ""

    keywords = "" + listKeyChannel + " " + listKeyTitle + " " + listKeyDesc + " " + getTarget(
        test_set.iloc[i, 6])

    listTotSplit = keywords.split()
    listTot = " | ".join(sorted(set(listTotSplit), key=listTotSplit.index))

    print(listTot)

    keywords_test.append(listTot)
    i += 1

test_df = pd.DataFrame({'keywords':keywords_test, 'rule':val_set['desc'], 'text':test_set['motivation']})

###############################################################################
# Extract candidates from validation data
###############################################################################

keywords_val = []
i = 0
while i < len(val_set):
    channel = "if " + ftfy.fix_text(val_set.iloc[i, 0]) + " then " + ftfy.fix_text(val_set.iloc[i, 2])
    title = ftfy.fix_text(val_set.iloc[i, 4])
    desc = ftfy.fix_text(val_set.iloc[i, 5])

    try:
        listKeyChannel = keyWordOperation(channel, n_keywords, n_gram_range, stop_words) + " " + ftfy.fix_text(
            training_set.iloc[i, 1]) + " " + ftfy.fix_text(training_set.iloc[i, 3])
    except:
        listKeyChannel = ""

    try:
        listKeyTitle = keyWordOperation(title, n_keywords, n_gram_range, stop_words)
    except:
        listKeyTitle = ""

    try:
        listKeyDesc = keyWordOperation(desc, n_keywords, n_gram_range, stop_words)
    except:
        listKeyDesc = ""

    keywords = "" + listKeyChannel + " " + listKeyTitle + " " + listKeyDesc + " " + getTarget(
        val_set.iloc[i, 6])

    listTotSplit = keywords.split()
    listTot = " | ".join(sorted(set(listTotSplit), key=listTotSplit.index))

    print(listTot)

    keywords_val.append(listTot)
    i += 1

val_df = pd.DataFrame({'keywords':keywords_val, 'rule':val_set['desc'], 'text':val_set['motivation']})

if torch.cuda.is_available():
   dev = torch.device("cuda:0")
   print("Running on the GPU")
else:
   dev = torch.device("cpu")
   print("Running on the CPU")

###############################################################################
# Load model
###############################################################################

tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base', return_dict=True)
#moving the model to GPU
model.to(dev)

optimizer = Adafactor(model.parameters(),lr=1e-3,
                      eps=(1e-30, 1e-3),
                      clip_threshold=1.0,
                      decay_rate=-0.8,
                      beta1=None,
                      weight_decay=0.0,
                      relative_step=False,
                      scale_parameter=False,
                      warmup_init=False)

# Sets the model in training mode
model.train()

num_of_epochs = 30
batch_size = 10
num_of_batches= int(len(train_df)/batch_size)

###############################################################################
# Training code
###############################################################################

loss_per_10_steps = []
for epoch in range(1, num_of_epochs + 1):
    print('Running epoch: {}'.format(epoch))

    running_loss = 0

    for i in range(num_of_batches):
        inputbatch = []
        labelbatch = []
        new_df = train_df[i * batch_size:i * batch_size + batch_size]
        for indx, row in new_df.iterrows():
            input = 'justification generation: ' + row['keywords'] + '</s>'
            labels = row['text'].split('harm')[1] + '</s>'
            inputbatch.append(input)
            labelbatch.append(labels)
        inputbatch = tokenizer.batch_encode_plus(inputbatch, padding=True, max_length=30, return_tensors='pt')[
            "input_ids"]
        labelbatch = tokenizer.batch_encode_plus(labelbatch, padding=True, max_length=30, return_tensors="pt")[
            "input_ids"]
        inputbatch = inputbatch.to(dev)
        labelbatch = labelbatch.to(dev)

        # clear out the gradients of all Variables
        optimizer.zero_grad()

        # Forward propogation
        outputs = model(input_ids=inputbatch, labels=labelbatch)
        loss = outputs.loss
        loss_num = loss.item()
        logits = outputs.logits
        running_loss += loss_num
        if i % 10 == 0:
            loss_per_10_steps.append(loss_num)

        # calculating the gradients
        loss.backward()

        # updating the params
        optimizer.step()

    running_loss = running_loss / int(num_of_batches)
    print('Epoch: {} , Running loss: {}'.format(epoch, running_loss))

###############################################################################
# Save model
###############################################################################

model_path = os.path.join('./model_4', 'model.pt')

with open(model_path, 'wb') as f:
    torch.save(model, f)

###############################################################################
# Load model
###############################################################################

tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base', return_dict=True)

with open(model_path, 'rb') as f:
    model = torch.load(f).to(dev)

###############################################################################
# Run on test data
###############################################################################

gold_motivations = test_df['text']
generated = []

for i in range(0, len(test_df)):
    keywords=[test_df['keywords'][i]]
    print("Keywords extracted: ", keywords)
    new_motivation = generate(keywords, model, tokenizer)
    generated.append(new_motivation)
    print("New motivation: ", new_motivation)

test_set['motivation_generated'] = generated
test_set.to_csv('test_set_results_T5_KeyToText_evaluated_4.csv')



