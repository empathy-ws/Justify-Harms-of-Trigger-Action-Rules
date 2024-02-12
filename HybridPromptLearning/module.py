from transformers import PhiForCausalLM, GPTNeoXForCausalLM, GPT2LMHeadModel
import torch
import torch.nn as nn

class PhiPrompt:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, weights, freezeLM=True, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # freeze pretrained model parameters
        if freezeLM:
            for param in model.parameters():
                param.requires_grad = False

        model.init_prompt(weights)
        return model

    def init_prompt(self, weights):
        self.src_len = 22
        embeddings = torch.FloatTensor(weights)

        self.trigger_service_embeddings = nn.Embedding.from_pretrained(embeddings)
        self.action_service_embeddings = nn.Embedding.from_pretrained(embeddings)

    def forward(self, user, item, context, explanation, mask, ignore_index=-100):
        device = user.device

        # embeddings
        u_src = self.trigger_service_embeddings(user)  # (batch_size, emsize)
        i_src = self.action_service_embeddings(item)  # (batch_size, emsize)
        text = torch.cat([context, explanation], 1)
        text = self.model.embed_tokens(text)
        src = torch.cat([u_src.unsqueeze(1), i_src.unsqueeze(1), text], 1)  # (batch_size, total_len, emsize)

        if mask is None:
            # auto-regressive generation
            return super().forward(inputs_embeds=src)
        else:
            # training
            # input padding
            pad_left = torch.ones((user.size(0), self.src_len), dtype=torch.int64).to(device)
            pad_input = torch.cat([pad_left, mask], 1)  # (batch_size, total_len)

            # prediction for training
            pred_left = torch.full((user.size(0), self.src_len), ignore_index, dtype=torch.int64).to(device)  # (batch_size, src_len)
            pred_right = torch.where(mask == 1, explanation, torch.tensor(ignore_index).to(device))  # replace <pad> with ignore_index
            prediction = torch.cat([pred_left, pred_right], 1)  # (batch_size, total_len)

            print('Fatto')

            return super().forward(attention_mask=pad_input, inputs_embeds=src, labels=prediction)

class PhiPromptLearning(PhiPrompt, PhiForCausalLM):
    def __init__(self, config):
        super().__init__(config)

################################################################  

class PythiaPrompt:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, weights, freezeLM=True, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # freeze pretrained model parameters
        if freezeLM:
            for param in model.parameters():
                param.requires_grad = False

        model.init_prompt(weights)
        return model

    def init_prompt(self, weights):
        self.src_len = 22
        embeddings = torch.FloatTensor(weights)

        self.trigger_service_embeddings = nn.Embedding.from_pretrained(embeddings)
        self.action_service_embeddings = nn.Embedding.from_pretrained(embeddings)

    def forward(self, user, item, context, explanation, mask, ignore_index=-100):
        device = user.device

        # embeddings
        u_src = self.trigger_service_embeddings(user)  # (batch_size, emsize)
        i_src = self.action_service_embeddings(item)  # (batch_size, emsize)
        text = torch.cat([context, explanation], 1)
        text = self.gpt_neox.embed_in(text)

        src = torch.cat([u_src.unsqueeze(1), i_src.unsqueeze(1), text], 1)  # (batch_size, total_len, emsize)

        if mask is None:
            # auto-regressive generation
            return super().forward(inputs_embeds=src)
        else:
            # training
            # input padding
            pad_left = torch.ones((user.size(0), self.src_len), dtype=torch.int64).to(device)
            pad_input = torch.cat([pad_left, mask], 1)  # (batch_size, total_len)

            # prediction for training
            pred_left = torch.full((user.size(0), self.src_len), ignore_index, dtype=torch.int64).to(device)  # (batch_size, src_len)
            pred_right = torch.where(mask == 1, explanation, torch.tensor(ignore_index).to(device))  # replace <pad> with ignore_index
            prediction = torch.cat([pred_left, pred_right], 1)  # (batch_size, total_len)

            print('Fatto')

            return super().forward(attention_mask=pad_input, inputs_embeds=src, labels=prediction)

class PythiaPromptLearning(PythiaPrompt, GPTNeoXForCausalLM):
    def __init__(self, config):
        super().__init__(config)

################################################################
        

class GPT2Prompt:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, weights, freezeLM=True, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # freeze pretrained model parameters
        if freezeLM:
            for param in model.parameters():
                param.requires_grad = False

        model.init_prompt(weights)
        return model

    def init_prompt(self, weights):
        self.src_len = 22
        embeddings = torch.FloatTensor(weights)

        self.trigger_service_embeddings = nn.Embedding.from_pretrained(embeddings)
        self.action_service_embeddings = nn.Embedding.from_pretrained(embeddings)

    def forward(self, user, item, context, explanation, mask, ignore_index=-100):
        device = user.device

        # embeddings
        u_src = self.trigger_service_embeddings(user)  # (batch_size, emsize)
        i_src = self.action_service_embeddings(item)  # (batch_size, emsize)
        text = torch.cat([context, explanation], 1)
        text = self.transformer.wte(text)
        src = torch.cat([u_src.unsqueeze(1), i_src.unsqueeze(1), text], 1)  # (batch_size, total_len, emsize)

        if mask is None:
            # auto-regressive generation
            return super().forward(inputs_embeds=src)
        else:
            # training
            # input padding
            pad_left = torch.ones((user.size(0), self.src_len), dtype=torch.int64).to(device)
            pad_input = torch.cat([pad_left, mask], 1)  # (batch_size, total_len)

            # prediction for training
            pred_left = torch.full((user.size(0), self.src_len), ignore_index, dtype=torch.int64).to(device)  # (batch_size, src_len)
            pred_right = torch.where(mask == 1, explanation, torch.tensor(ignore_index).to(device))  # replace <pad> with ignore_index
            prediction = torch.cat([pred_left, pred_right], 1)  # (batch_size, total_len)

            print('Fatto')

            return super().forward(attention_mask=pad_input, inputs_embeds=src, labels=prediction)

class GPT2PromptLearning(GPT2Prompt, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)