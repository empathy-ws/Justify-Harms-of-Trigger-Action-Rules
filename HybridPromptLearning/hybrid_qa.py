# Prompt+LM Tuning

import os
import math
import torch
import argparse
import pandas as pd
import ftfy
from module import HybridPromptLearning
from transformers import GPT2Tokenizer, AdamW
from utils import Batchify4, now_time, ids2tokens


parser = argparse.ArgumentParser(description='Hybrid Prompt Learning for Generating Justifications of Security Risks in Automation Rules')
parser.add_argument('--train_path', type=str, default="./training_dataset.csv")
parser.add_argument('--test_path', type=str, default="./test_dataset.csv")
parser.add_argument('--val_path', type=str, default="./val_dataset.csv")
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate for the model')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log_interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--checkpoint', type=str, default='./unisa/',
                    help='directory to save the final model')
parser.add_argument('--outf', type=str, default='generated_prompt/generated.txt',
                    help='output file for generated text')
parser.add_argument('--endure_times', type=int, default=25,
                    help='the maximum endure times of loss increasing on validation')
parser.add_argument('--words', type=int, default=20,
                    help='number of words to generate for each sample')
args = parser.parse_args()

print('-' * 40 + 'ARGUMENTS' + '-' * 40)
for arg in vars(args):
    print('{:40} {}'.format(arg, getattr(args, arg)))
print('-' * 40 + 'ARGUMENTS' + '-' * 40)

if torch.cuda.is_available():
    if not args.cuda:
        print(now_time() + 'WARNING: You have a CUDA device, so you should probably run with --cuda')
device = torch.device('cuda' if args.cuda else 'cpu')

if not os.path.exists(args.checkpoint):
    os.makedirs(args.checkpoint)
model_path = os.path.join(args.checkpoint, 'model_hybrid.pt')
prediction_path = os.path.join(args.checkpoint, args.outf)

###############################################################################
# Load data
###############################################################################

print(now_time() + 'Loading data')
bos = '<bos>'
eos = '<eos>'
pad = '<pad>'
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token=bos, eos_token=eos, pad_token=pad)
col_names = ['triggerTitle','triggerChannelTitle','actionTitle','actionChannelTitle','title','desc','target','justification']

train_df = pd.read_csv(args.train_path,skiprows=1,sep=';',names=col_names,encoding = "ISO-8859-1")
test_df = pd.read_csv(args.test_path,skiprows=1,sep=';',names=col_names,encoding = "ISO-8859-1")
val_df = pd.read_csv(args.val_path,skiprows=1,sep=';',names=col_names,encoding = "ISO-8859-1")

service_df = pd.read_csv('skip_gram/service_weights_90.csv', sep=',', encoding ="ISO-8859-1")

weights = []
word2id = {}

for i in range(len(service_df)):
    word2id[ftfy.fix_text(service_df.loc[i][0]).strip()] = i
    weights.append(service_df.loc[i][1:len(service_df.loc[i])])

train_data = Batchify4(train_df, tokenizer, bos, eos, args.words, word2id, args.batch_size, shuffle=True)
val_data = Batchify4(val_df, tokenizer, bos, eos, args.words, word2id, args.batch_size)
test_data = Batchify4(test_df, tokenizer, bos, eos, args.words, word2id, args.batch_size)

###############################################################################
# Build the model
###############################################################################

vocab_size = len(service_df)
ntoken = len(tokenizer)
model = HybridPromptLearning.from_pretrained('gpt2', weights)
model.resize_token_embeddings(ntoken)  # three tokens added, update embedding table
model.to(device)

###############################################################################
# Training code
###############################################################################

def train(data):
    # Turn on training mode which enables dropout.
    model.train()
    text_loss = 0.
    total_sample = 0
    while True:
        trigger_service, action_service, prompt, seq, mask = data.next_batch()  # data.step += 1
        trigger_service = trigger_service.to(device)  # (batch_size,)
        action_service = action_service.to(device)
        prompt = prompt.to(device)
        seq = seq.to(device)  # (batch_size, seq_len)
        mask = mask.to(device)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad()
        outputs = model(trigger_service, action_service, prompt, seq, mask)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        batch_size = trigger_service.size(0)
        text_loss += batch_size * loss.item()
        total_sample += batch_size

        if data.step % args.log_interval == 0 or data.step == data.total_step:
            cur_t_loss = text_loss / total_sample
            print(now_time() + 'text ppl {:4.4f} | {:5d}/{:5d} batches'.format(math.exp(cur_t_loss), data.step, data.total_step))
            text_loss = 0.
            total_sample = 0
        if data.step == data.total_step:
            break

def evaluate(data):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    text_loss = 0.
    total_sample = 0
    with torch.no_grad():
        while True:
            trigger_service, action_service, prompt, seq, mask = data.next_batch()  # data.step += 1
            trigger_service = trigger_service.to(device)  # (batch_size,)
            action_service = action_service.to(device)
            prompt = prompt.to(device)
            seq = seq.to(device)  # (batch_size, seq_len)
            mask = mask.to(device)
            outputs = model(trigger_service, action_service, prompt, seq, mask)
            loss = outputs.loss

            batch_size = trigger_service.size(0)
            text_loss += batch_size * loss.item()
            total_sample += batch_size

            if data.step == data.total_step:
                break
    return text_loss / total_sample


def generate(data):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    idss_predict = []
    with torch.no_grad():
        while True:
            trigger_service, action_service, prompt, seq, _ = data.next_batch()  # data.step += 1
            trigger_service = trigger_service.to(device)  # (batch_size,)
            action_service = action_service.to(device)
            prompt = prompt.to(device)
            text = seq[:, :1].to(device)  # bos, (batch_size, 1)
            for idx in range(seq.size(1)):
                # produce a word at each step
                outputs = model(trigger_service, action_service, prompt, text, None)
                last_token = outputs.logits[:, -1, :]  # the last token, (batch_size, ntoken)
                word_prob = torch.softmax(last_token, dim=-1)
                token = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1), pick the one with the largest probability
                text = torch.cat([text, token], 1)  # (batch_size, len++)
            ids = text[:, 1:].tolist()  # remove bos, (batch_size, seq_len)
            idss_predict.extend(ids)

            if data.step == data.total_step:
                break
    return idss_predict

print(now_time() + 'Tuning both Prompt and LM')
for param in model.parameters():
    param.requires_grad = True
optimizer = AdamW(model.parameters(), lr=args.lr, no_deprecation_warning=True)

# Loop over epochs.
best_val_loss = float('inf')
endure_count = 0
for epoch in range(1, args.epochs + 1):
    print(now_time() + 'epoch {}'.format(epoch))
    train(train_data)
    val_loss = evaluate(val_data)
    print(now_time() + 'text ppl {:4.4f} | valid loss {:4.4f} on validation'.format(math.exp(val_loss), val_loss))
    # Save the model if the validation loss is the best we've seen so far.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        with open(model_path, 'wb') as f:
            torch.save(model, f)
    else:
        endure_count += 1
        print(now_time() + 'Endured {} time(s)'.format(endure_count))
        if endure_count == args.endure_times:
            print(now_time() + 'Cannot endure it anymore | Exiting from early stop')
            break

# Load the best saved model.
with open(model_path, 'rb') as f:
    model = torch.load(f).to(device)

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print(now_time() + 'text ppl {:4.4f} on test | End of training'.format(math.exp(test_loss)))
print(now_time() + 'Generating text')
idss_predicted = generate(test_data)
tokens_test = [ids2tokens(ids[1:], tokenizer, eos) for ids in test_data.seq.tolist()]
tokens_predict = [ids2tokens(ids, tokenizer, eos) for ids in idss_predicted]
text_test = [' '.join(tokens) for tokens in tokens_test]
text_predict = [' '.join(tokens) for tokens in tokens_predict]
print(text_test)
print(text_predict)
with open(prediction_path, 'w', encoding='utf-8') as fp:
    for justification in text_predict:
        fp.write("%s\n" % justification)
print(now_time() + 'Generated text saved to ({})'.format(prediction_path))
