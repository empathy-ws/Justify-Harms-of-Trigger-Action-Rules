# Prompt+LM Tuning

import os
import torch
import argparse
import pandas as pd
import ftfy
from module import PhiPromptLearning
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import now_time, ids2tokens

model_size = '1_5_dev'
threshold = '85'
embedding_size = '2048'

parser = argparse.ArgumentParser(description='Justify Harms of Trigger Action Rules')
parser.add_argument('--train_path', type=str, default="./training_dataset.csv")
parser.add_argument('--test_path', type=str, default="./test_dataset.csv")
parser.add_argument('--val_path', type=str, default="./val_dataset.csv")
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate for the model')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=60,
                    help='batch size')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log_interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--checkpoint', type=str, default='./unisa/',
                    help='directory to save the final model')
parser.add_argument('--outf', type=str, default='generated_prompt/generated_phi_span_' + threshold + '_' + model_size + '.txt',
                    help='output file for generated text')
parser.add_argument('--endure_times', type=int, default=3,
                    help='the maximum endure times of loss increasing on validation')
parser.add_argument('--words', type=int, default=20,
                    help='number of words to generate for each sample')
args = parser.parse_args()

print('-' * 40 + 'ARGUMENTS' + '-' * 40)
for arg in vars(args):
    print('{:40} {}'.format(arg, getattr(args, arg)))
print('-' * 40 + 'ARGUMENTS' + '-' * 40)

device = torch.device('cuda')

if not os.path.exists(args.checkpoint):
    os.makedirs(args.checkpoint)
model_path = os.path.join(args.checkpoint, 'phi_span_' + threshold + '_' + model_size + '.pt')
prediction_path = os.path.join(args.checkpoint, args.outf)

###############################################################################
# Load data
###############################################################################

print(now_time() + 'Loading data')
bos = '<bos>'
eos = '<eos>'
pad = '<pad>'
tokenizer = AutoTokenizer.from_pretrained("susnato/phi-" + model_size, bos_token=bos, eos_token=eos, pad_token=pad)
col_names = ['triggerTitle','triggerChannelTitle','actionTitle','actionChannelTitle','title','desc','target','motivation']

test_df = pd.read_csv(args.test_path,skiprows=1,sep=';',names=col_names,encoding = "ISO-8859-1")

service_df = pd.read_csv('skip_gram/service_weights_' + threshold + '_' + embedding_size + '.csv', sep=',', encoding ="ISO-8859-1")

weights = []
word2id = {}

for i in range(len(service_df)):
    word2id[ftfy.fix_text(service_df.loc[i][0]).strip()] = i
    weights.append(service_df.loc[i][1:len(service_df.loc[i])])

###############################################################################
# Build the model
###############################################################################

vocab_size = len(service_df)
ntoken = len(tokenizer)
model = PhiPromptLearning.from_pretrained("susnato/phi-" + model_size, weights)
model.resize_token_embeddings(ntoken)  # three tokens added, update embedding table
model.to(device)

def convert_class(prediction):
    if prediction == 1:
        return "Personal"
    elif prediction == 2:
        return "Physical"
    else:
        return "Cybersecurity"

def remove_eos(text):
    segments = text.split('<eos>')

    return segments[0].strip()

def generate_infilling_model(rule):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    idss_predict = []
    with torch.no_grad():
        explanation = '{} {} {}'.format(bos, ' ', eos)
        encoded_inputs = tokenizer(explanation, padding=True, return_tensors='pt')
        seq = encoded_inputs['input_ids'].contiguous().to(device)

        trigger_service = [word2id[ftfy.fix_text(rule[0]).strip()]]
        action_service = [word2id[ftfy.fix_text(rule[1]).strip()]]

        trigger_service = torch.tensor(trigger_service, dtype=torch.int64).contiguous().to(device)
        action_service = torch.tensor(action_service, dtype=torch.int64).contiguous().to(device)

        features = 'input=[' + 'triggerTitle=' + ftfy.fix_text(
            rule[2]) + '; actionTitle=' + ftfy.fix_text(
            rule[3]) + '; title=' + ftfy.fix_text(
            rule[4]) + '; desc=' + ftfy.fix_text(
            rule[5]) + '] ' + 'This rule might cause a ' + convert_class(rule[6]) + ' harm'

        encoded_features = tokenizer(features, padding=True, return_tensors='pt')
        prompt = encoded_features['input_ids'][:, :args.words].contiguous().to(device)
        prompt = prompt.to(device)

        text = seq[:, :1].to(device)  # bos, (batch_size, 1)
        for idx in range(100):
            ids_predict = []
            # produce a word at each step
            outputs = model(trigger_service, action_service, prompt, text, None)
            last_token = outputs.logits[:, -1, :]  # the last token, (batch_size, ntoken)
            word_prob = torch.softmax(last_token, dim=-1)
            token = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1), pick the one with the largest probability
            text = torch.cat([text, token], 1)  # (batch_size, len++)
            int_rep = text[:, 1:].tolist()
            ids_predict.extend(int_rep)
            tokens_predict = [ids2tokens(ids, tokenizer, eos) for ids in ids_predict]
            text_predict = [' '.join(tokens) for tokens in tokens_predict]
            if '<eos>' in text_predict[0]:
                ids = text[:, 1:].tolist()  # remove bos, (batch_size, seq_len)
                idss_predict.extend(ids)
                return idss_predict
        ids = text[:, 1:].tolist()  # remove bos, (batch_size, seq_len)
        idss_predict.extend(ids)
        return idss_predict

# Load the best saved model.
with open(model_path, 'rb') as f:
    model = torch.load(f).to(device)


print(now_time() + 'Generating text')

motivations = []

for i in tqdm(range(0, len(test_df))):
    rule = [test_df.iloc[i]['triggerChannelTitle'],
            test_df.iloc[i]['actionChannelTitle'],
            test_df.iloc[i]['triggerTitle'],
            test_df.iloc[i]['actionTitle'],
            test_df.iloc[i]['title'],
            test_df.iloc[i]['desc'],
            test_df.iloc[i]['target']]
    print(rule)
    idss_predicted = generate_infilling_model(rule)
    tokens_predict = [ids2tokens(ids, tokenizer, eos) for ids in idss_predicted]
    text_predict = [' '.join(tokens) for tokens in tokens_predict]
    motivation = "This rule might cause a " + convert_class(test_df.iloc[i]['target']) + " harm " + remove_eos(text_predict[0])
    print(motivation)
    motivations.append(motivation)

with open(prediction_path, 'w', encoding='utf-8') as fp:
    for motivation in motivations:
        fp.write("%s\n" % motivation)
print(now_time() + 'Generated text saved to ({})'.format(prediction_path))