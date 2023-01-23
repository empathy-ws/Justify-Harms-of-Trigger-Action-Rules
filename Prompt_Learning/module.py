from transformers import GPT2LMHeadModel
import torch
import torch.nn as nn

class DiscretePrompt:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    def forward(self, context, justification, exp_mask, ignore_index=-100):
        device = context.device
        text = torch.cat([context, justification], 1)  # (batch_size, total_len)
        src = self.transformer.wte(text)  # (batch_size, total_len, emsize)

        if exp_mask is None:
            # auto-regressive generation
            return super().forward(inputs_embeds=src)
        else:
            # training
            # input padding
            pad_left = torch.ones_like(context, dtype=torch.int64).to(device)
            pad_input = torch.cat([pad_left, exp_mask], 1)  # (batch_size, total_len)

            # prediction for training
            pred_left = torch.full_like(context, ignore_index, dtype=torch.int64).to(device)  # (batch_size, src_len)
            pred_right = torch.where(exp_mask == 1, justification, torch.tensor(ignore_index).to(device))  # replace <pad> with ignore_index
            prediction = torch.cat([pred_left, pred_right], 1)  # (batch_size, total_len)

            return super().forward(attention_mask=pad_input, inputs_embeds=src, labels=prediction)

class DiscretePromptLearning(DiscretePrompt, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)

class HybridPrompt:
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

    def forward(self, trigger_service, action_service, context, justification, mask, ignore_index=-100):
        device = trigger_service.device

        # embeddings
        t_src = self.trigger_service_embeddings(trigger_service)  # (batch_size, emsize)
        a_src = self.action_service_embeddings(action_service)  # (batch_size, emsize)
        text = torch.cat([context, justification], 1)
        text = self.transformer.wte(text)
        src = torch.cat([t_src.unsqueeze(1), a_src.unsqueeze(1), text], 1)  # (batch_size, total_len, emsize)

        if mask is None:
            # auto-regressive generation
            return super().forward(inputs_embeds=src)
        else:
            # training
            # input padding
            pad_left = torch.ones((trigger_service.size(0), self.src_len), dtype=torch.int64).to(device)
            pad_input = torch.cat([pad_left, mask], 1)  # (batch_size, total_len)

            # prediction for training
            pred_left = torch.full((trigger_service.size(0), self.src_len), ignore_index, dtype=torch.int64).to(device)  # (batch_size, src_len)
            pred_right = torch.where(mask == 1, justification, torch.tensor(ignore_index).to(device))  # replace <pad> with ignore_index
            prediction = torch.cat([pred_left, pred_right], 1)  # (batch_size, total_len)

            return super().forward(attention_mask=pad_input, inputs_embeds=src, labels=prediction)

class HybridPromptLearning(HybridPrompt, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
