import re
import math
import random
import datetime
import ftfy
import torch

def convert_class(prediction):
    if prediction == 1:
        return "Personal"
    elif prediction == 2:
        return "Physical"
    else:
        return "Cybersecurity"

class Batchify:
    def __init__(self, data, tokenizer, bos, eos, seq_len, word2id, batch_size=128, shuffle=False):
        trigger_services, action_services, t, features = [], [], [], []
        for i in range(0, len(data)):
            trigger_services.append(word2id[ftfy.fix_text(data.iloc[i]['triggerChannelTitle']).strip()])
            action_services.append(word2id[ftfy.fix_text(data.iloc[i]['actionChannelTitle']).strip()])
            features.append('input=[' + 'triggerTitle=' + ftfy.fix_text(
                data.iloc[i]['triggerTitle']) + '; actionTitle=' + ftfy.fix_text(
                data.iloc[i]['actionTitle']) + '; title=' + ftfy.fix_text(
                data.iloc[i]['title']) + '; desc=' + ftfy.fix_text(data.iloc[i]['desc']) + '] ' + 'This rule might cause a ' + convert_class(data.iloc[i]['target']) + ' harm')

            t.append('{} {} {}'.format(bos, data.iloc[i]['motivation'].split('harm')[1], eos))

        encoded_inputs = tokenizer(t, padding=True, return_tensors='pt')
        self.seq = encoded_inputs['input_ids'].contiguous()
        self.mask = encoded_inputs['attention_mask'].contiguous()
        encoded_features = tokenizer(features, padding=True, return_tensors='pt')
        self.prompt = encoded_features['input_ids'][:, :seq_len].contiguous()
        self.trigger_service = torch.tensor(trigger_services, dtype=torch.int64).contiguous()
        self.action_service = torch.tensor(action_services, dtype=torch.int64).contiguous()
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sample_num = len(data)
        self.index_list = list(range(self.sample_num))
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0

    def next_batch(self):
        if self.step == self.total_step:
            self.step = 0
            if self.shuffle:
                random.shuffle(self.index_list)

        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        index = self.index_list[start:offset]
        seq = self.seq[index]  # (batch_size, seq_len)
        mask = self.mask[index]
        prompt = self.prompt[index]
        trigger_service = self.trigger_service[index]  # (batch_size,)
        action_service = self.action_service[index]
        return trigger_service, action_service, prompt, seq, mask

class Batchify2:
    def __init__(self, data, tokenizer, bos, eos, seq_len, batch_size=128, shuffle=False):
        t, features = [], []
        for i in range(0, len(data)):
            features.append('input=[' + 'triggerTitle=' + ftfy.fix_text(data.iloc[i]['triggerTitle']) + '; triggerChannelTitle=' + ftfy.fix_text(data.iloc[i]['triggerChannelTitle']) + '; actionTitle=' + ftfy.fix_text(data.iloc[i]['actionTitle']) + '; actionChannelTitle=' + ftfy.fix_text(data.iloc[i]['actionChannelTitle']) + '; title=' + ftfy.fix_text(data.iloc[i]['title']) + '; desc=' + ftfy.fix_text(data.iloc[i]['desc']) + '] ' +
                            'This rule might cause a ' + convert_class(data.iloc[i]['target']) + ' harm')

            t.append('{} {} {}'.format(bos, data.iloc[i]['motivation'].split('harm')[1], eos))

        encoded_inputs = tokenizer(t, padding=True, return_tensors='pt')
        self.seq = encoded_inputs['input_ids'].contiguous()
        self.mask = encoded_inputs['attention_mask'].contiguous()
        encoded_features = tokenizer(features, padding=True, return_tensors='pt')
        self.prompt = encoded_features['input_ids'][:, :seq_len].contiguous()
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sample_num = len(data)
        self.index_list = list(range(self.sample_num))
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0

    def next_batch(self):
        if self.step == self.total_step:
            self.step = 0
            if self.shuffle:
                random.shuffle(self.index_list)

        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        index = self.index_list[start:offset]
        seq = self.seq[index]  # (batch_size, seq_len)
        mask = self.mask[index]
        prompt = self.prompt[index]
        return seq, mask, prompt

class Batchify3:
    def __init__(self, data, tokenizer, bos, eos, seq_len, batch_size=128, shuffle=False):
        t, features = [], []
        for i in range(0, len(data)):
            features.append('Why might rule ' + 'input=[' + 'triggerTitle=' + ftfy.fix_text(data.iloc[i]['triggerTitle']) + '; triggerChannelTitle=' + ftfy.fix_text(data.iloc[i]['triggerChannelTitle']) + '; actionTitle=' + ftfy.fix_text(data.iloc[i]['actionTitle']) + '; actionChannelTitle=' + ftfy.fix_text(data.iloc[i]['actionChannelTitle']) + '; title=' + ftfy.fix_text(data.iloc[i]['title']) + '; desc=' + ftfy.fix_text(data.iloc[i]['desc']) + ']' +
                            'cause ' + convert_class(data.iloc[i]['target']) + ' harm? ')

            t.append('{} {} {}'.format(bos, data.iloc[i]['motivation'].split('harm')[1], eos))

        encoded_inputs = tokenizer(t, padding=True, return_tensors='pt')
        self.seq = encoded_inputs['input_ids'].contiguous()
        self.mask = encoded_inputs['attention_mask'].contiguous()
        encoded_features = tokenizer(features, padding=True, return_tensors='pt')
        self.prompt = encoded_features['input_ids'][:, :seq_len].contiguous()
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sample_num = len(data)
        self.index_list = list(range(self.sample_num))
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0

    def next_batch(self):
        if self.step == self.total_step:
            self.step = 0
            if self.shuffle:
                random.shuffle(self.index_list)

        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        index = self.index_list[start:offset]
        seq = self.seq[index]  # (batch_size, seq_len)
        mask = self.mask[index]
        prompt = self.prompt[index]
        return seq, mask, prompt

class Batchify4:
    def __init__(self, data, tokenizer, bos, eos, seq_len, word2id, batch_size=128, shuffle=False):
        trigger_services, action_services, t, features = [], [], [], []
        for i in range(0, len(data)):
            trigger_services.append(word2id[ftfy.fix_text(data.iloc[i]['triggerChannelTitle']).strip()])
            action_services.append(word2id[ftfy.fix_text(data.iloc[i]['actionChannelTitle']).strip()])
            features.append('Why might rule ' + 'input=[' + 'triggerTitle=' + ftfy.fix_text(
                data.iloc[i]['triggerTitle']) + '; actionTitle=' + ftfy.fix_text(
                data.iloc[i]['actionTitle']) + '; title=' + ftfy.fix_text(
                data.iloc[i]['title']) + '; desc=' + ftfy.fix_text(data.iloc[i]['desc']) + ']' +
                            'cause ' + convert_class(data.iloc[i]['target']) + ' harm? ')

            t.append('{} {} {}'.format(bos, data.iloc[i]['motivation'].split('harm')[1], eos))

        encoded_inputs = tokenizer(t, padding=True, return_tensors='pt')
        self.seq = encoded_inputs['input_ids'].contiguous()
        self.mask = encoded_inputs['attention_mask'].contiguous()
        encoded_features = tokenizer(features, padding=True, return_tensors='pt')
        self.prompt = encoded_features['input_ids'][:, :seq_len].contiguous()
        self.trigger_service = torch.tensor(trigger_services, dtype=torch.int64).contiguous()
        self.action_service = torch.tensor(action_services, dtype=torch.int64).contiguous()
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sample_num = len(data)
        self.index_list = list(range(self.sample_num))
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0

    def next_batch(self):
        if self.step == self.total_step:
            self.step = 0
            if self.shuffle:
                random.shuffle(self.index_list)

        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        index = self.index_list[start:offset]
        seq = self.seq[index]  # (batch_size, seq_len)
        mask = self.mask[index]
        prompt = self.prompt[index]
        trigger_service = self.trigger_service[index]  # (batch_size,)
        action_service = self.action_service[index]
        return trigger_service, action_service, prompt, seq, mask

def now_time():
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '


def postprocessing(string):
    string = re.sub('\'s', ' \'s', string)
    string = re.sub('\'m', ' \'m', string)
    string = re.sub('\'ve', ' \'ve', string)
    string = re.sub('n\'t', ' n\'t', string)
    string = re.sub('\'re', ' \'re', string)
    string = re.sub('\'d', ' \'d', string)
    string = re.sub('\'ll', ' \'ll', string)
    string = re.sub('\(', ' ( ', string)
    string = re.sub('\)', ' ) ', string)
    string = re.sub(',+', ' , ', string)
    string = re.sub(':+', ' , ', string)
    string = re.sub(';+', ' . ', string)
    string = re.sub('\.+', ' . ', string)
    string = re.sub('!+', ' ! ', string)
    string = re.sub('\?+', ' ? ', string)
    string = re.sub(' +', ' ', string).strip()
    return string


def ids2tokens(ids, tokenizer, eos):
    text = tokenizer.decode(ids)
    text = postprocessing(text)  # process punctuations: "good!" -> "good !"
    tokens = []
    for token in text.split():
        if token == eos:
            break
        tokens.append(token)
    return tokens
