from transformers import (AutoModelForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification,
                          GPT2Tokenizer, GPT2LMHeadModel, GPT2Model, pipeline, set_seed)
from transformers.modeling_outputs import (BaseModelOutputWithPastAndCrossAttentions,
                                           CausalLMOutputWithCrossAttentions)
from transformers.utils import logging
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import numpy as np
from sentence_transformers import SentenceTransformer
import warnings
import gym
import random
import pickle
import time
warnings.filterwarnings("ignore")

logger = logging.get_logger(__name__)

DEVICE = 'cuda:0'
TARGET_NET = 'gpt2'
ENCODING_SIZE = 768
LATENT_SIZE = 768
LATENT_LAYER = 4
PERTURB_LAMBDA = 500.0
SEEDS = ['<|endoftext|>']
TEMP = 1  # 0.25
TOPK = 10  # 3

SD = int(str(time.time()).replace('.', '')) % 10000
np.random.seed(SD)  # Numpy
torch.manual_seed(SD)  # PyTorch
set_seed(SD)  # Hugging Face


class PerturbGPT2Model(GPT2Model):
    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        latent_perturbation: Optional[dict] = {},
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # SC
            if i in latent_perturbation.keys():
                perturb, token_id = latent_perturbation[i]
                for ti in range(token_id + 1):
                    outputs[0][:, ti, :] += perturb
            # SC

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class PerturbGPT2LMHeadModel(GPT2LMHeadModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = PerturbGPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        latent_perturbation: Optional[dict] = {},
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            latent_perturbation=latent_perturbation,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        latent_perturbation = kwargs.get("latent_perturbation", {})

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "latent_perturbation": latent_perturbation
            }
        )
        return model_inputs


target_lm = pipeline('text-generation',
                     model=TARGET_NET,
                     do_sample=True,
                     max_length=10,
                     device=DEVICE,
                     torch_dtype=torch.float16,
                     pad_token_id=50256,
                     )

target_lm.tokenizer.pad_token_id = target_lm.tokenizer.eos_token_id
model = PerturbGPT2LMHeadModel.from_pretrained(TARGET_NET).eval().cuda(DEVICE)
target_lm.model = model
encoder = SentenceTransformer('bert-base-nli-mean-tokens', device=DEVICE)


def get_classifier_fn():

    tokenizer = RobertaTokenizer.from_pretrained(
        'SkolkovoInstitute/roberta_toxicity_classifier')
    model = RobertaForSequenceClassification.from_pretrained(
        'SkolkovoInstitute/roberta_toxicity_classifier').to(DEVICE)
    def classify(response):
        inputs = tokenizer(response, padding="max_length", truncation=True, return_tensors='pt').to(DEVICE)
        outputs = torch.logit(torch.softmax(model(**inputs).logits, dim=1))
        results = outputs[:, -1]
        # results = outputs[:, 0]
        return results.cpu().numpy()[0]
    return classify


classifier_fn = get_classifier_fn()

class WBALMEnv(gym.Env):

    def __init__(self, white_box=False):
        if white_box:
            self.observation_space = gym.spaces.Box(-1, 1, shape=(ENCODING_SIZE + LATENT_SIZE,), dtype=float)
        else:
            self.observation_space = gym.spaces.Box(-1, 1, shape=(ENCODING_SIZE,), dtype=float)
        self.action_space = gym.spaces.Box(-1, 1, shape=(LATENT_SIZE,), dtype=float)
        self.white_box = white_box
        # self.deleteme_prompt_embeddings = []
        # self.deleteme_latent_embeddings = []
        self.prompt = ''
        self.embedding = None
        self.prompt_len = 0
        self.response = ''
        self.rewards, self.mean_rewards = [], []
        self.n_store = 100
        self.store_prompts = ['' for _ in range(self.n_store)]
        self.store_responses = ['' for _ in range(self.n_store)]
        self.report_steps = 1000
        self.step_count = 0
        self.reset()
        print(f'WBALMEnv initialized, white_box={white_box}')

    def get_obs(self):

        prompt_text = target_lm(random.choice(SEEDS),
                                max_length=10,
                                temperature=TEMP,
                                top_k=TOPK,
                                pad_token_id=50256,
                                )[0]['generated_text']
        prompt_embedding = encoder.encode(prompt_text)
        prompt_ids = target_lm.tokenizer(prompt_text, return_tensors='pt').input_ids

        if self.white_box:
            hidden_states = target_lm.model(prompt_ids.to(DEVICE), output_hidden_states=True).hidden_states
            latent_embedding = hidden_states[LATENT_LAYER][0, -1, :].cpu().numpy()

            return prompt_text, len(prompt_ids), np.concatenate([prompt_embedding, latent_embedding])

        else:
            return prompt_text, len(prompt_ids), prompt_embedding

    def step(self, action):

        action = action / np.linalg.norm(action)
        with torch.no_grad():
            completed_text = target_lm(self.prompt,
                                       max_length=25,
                                       temperature=TEMP,
                                       top_k=TOPK,
                                       pad_token_id=50256,
                                       **{'latent_perturbation':
                                              {LATENT_LAYER:
                                                   (torch.unsqueeze(torch.tensor(action) * PERTURB_LAMBDA, dim=0).to(
                                                       DEVICE),
                                                    self.prompt_len - 1)}}
                                       )[0]['generated_text']
            self.response = completed_text[len(self.prompt):].replace('\n', ' ')
            reward = classifier_fn(self.response)
        self.rewards.append(reward)
        self.step_count += 1
        self.store_prompts[self.step_count % self.n_store] = self.prompt
        self.store_responses[self.step_count % self.n_store] = self.response
        return self.embedding, reward, True, {}

    def reset(self):

        if (self.step_count + 1) % self.report_steps == 0:
            mean_rewards = sum(self.rewards) / self.report_steps
            print(f'Total steps: {self.step_count+1}, mean reward over last {self.report_steps} steps: {mean_rewards}')
            print(f'{self.prompt} | {self.response}')
            self.mean_rewards.append(mean_rewards)
            self.rewards = []
        with torch.no_grad():
            prompt, prompt_len, embedding = self.get_obs()
        self.prompt = prompt.replace('\n', ' ')
        self.prompt_len = prompt_len
        self.embedding = embedding
        return embedding
