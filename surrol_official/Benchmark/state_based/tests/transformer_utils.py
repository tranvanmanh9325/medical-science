import numpy as np
import torch
import torch.nn as nn

# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT-2 model."""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
)
from transformers.modeling_utils import (
    Conv1D,
    PreTrainedModel,
    SequenceSummary,
    find_pruneable_heads_and_indices,
    prune_conv1d_layer,
)
from transformers.utils import logging
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
import transformers

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "GPT2Config"
_TOKENIZER_FOR_DOC = "GPT2Tokenizer"

GPT2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "distilgpt2",
    # See all GPT-2 models at https://huggingface.co/models?filter=gpt2
]


def load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path):
    """Load tf checkpoints in a pytorch model"""
    try:
        import re

        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(gpt2_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array.squeeze())

    for name, array in zip(names, arrays):
        name = name[6:]  # skip "model/"
        name = name.split("/")
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                scope_names = re.split(r"(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "w" or scope_names[0] == "g":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "b":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "wpe" or scope_names[0] == "wte":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, scope_names[0])
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        try:
            assert (
                    pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False, is_cross_attention=False):
        super().__init__()

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer(
            "bias", torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx)
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.is_cross_attention = is_cross_attention
        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * n_state, nx)
            self.q_attn = Conv1D(n_state, nx)
        else:
            self.c_attn = Conv1D(3 * n_state, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_head, self.split_size // self.n_head, self.pruned_heads
        )
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / (float(v.size(-1)) ** 0.5)
        nd, ns = w.size(-2), w.size(-1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            mask = self.bias[:, :, ns - nd: ns, :ns]
            w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [torch.matmul(w, v)]
        if output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(
            self,
            hidden_states,
            layer_past=None,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            output_attentions=False,
    ):
        if encoder_hidden_states is not None:
            assert hasattr(
                self, "q_attn"
            ), "If class is used as cross attention, the weights `q_attn` have to be defined. Please make sure to instantiate class with `Attention(..., is_cross_attention=True)`."
            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        else:
            present = (None,)

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class AdapterMLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super().__init__()
        hidden_size = config.n_embd
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = Attention(hidden_size, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        # self.adapter_ln = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        if config.add_cross_attention:
            self.crossattention = Attention(hidden_size, n_ctx, config, scale, is_cross_attention=True)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = MLP(inner_dim, config)
        # self.adapter_mlp = AdapterMLP(512, config)  # ADAPTER

    def forward(
            self,
            hidden_states,
            layer_past=None,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            output_attentions=False,
    ):
        attn_outputs = self.attn(
            self.ln_1(hidden_states),
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + hidden_states

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            cross_attn_outputs = self.crossattention(
                self.ln_cross_attn(hidden_states),
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = hidden_states + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        feed_forward_hidden_states = self.mlp(self.ln_2(hidden_states))
        # residual connection
        hidden_states = hidden_states + feed_forward_hidden_states
        # hidden_states = hidden_states + self.adapter_ln(self.adapter_mlp(hidden_states))

        outputs = [hidden_states] + outputs
        return outputs  # hidden_states, present, (attentions, cross_attentions)


class GPT2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPT2Config
    load_tf_weights = load_tf_weights_in_gpt2
    base_model_prefix = "transformer"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            # module.weight.data.fill_(.01)  # KL: Adapter change


@dataclass
class GPT2DoubleHeadsModelOutput(ModelOutput):
    """
    Base class for outputs of models predicting if two sentences are consecutive or not.
    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided):
            Language modeling loss.
        mc_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`mc_labels` is provided):
            Multiple choice classification loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        mc_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices)`):
            Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
        past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`, with each tensor of shape :obj:`(2,
            batch_size, num_heads, sequence_length, embed_size_per_head)`).
            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            :obj:`past_key_values` input) to speed up sequential decoding.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    mc_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mc_logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


GPT2_START_DOCSTRING = r"""
    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)
    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.
    Parameters:
        config (:class:`~transformers.GPT2Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

GPT2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, input_ids_length)`):
            :obj:`input_ids_length` = ``sequence_length`` if :obj:`past_key_values` is ``None`` else
            ``past_key_values[0].shape[-2]`` (``sequence_length`` of input past key value states). Indices of input
            sequence tokens in the vocabulary.
            If :obj:`past_key_values` is used, only ``input_ids`` that do not have their past calculated should be
            passed as ``input_ids``.
            Indices can be obtained using :class:`~transformers.GPT2Tokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.
            `What are input IDs? <../glossary.html#input-ids>`__
        past_key_values (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`):
            Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see
            :obj:`past_key_values` output below). Can be used to speed up sequential decoding. The ``input_ids`` which
            have their past given to this model should not be passed as ``input_ids`` as they have already been
            computed.
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, input_ids_length)`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:
            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.
            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.
            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
            If :obj:`past_key_values` is used, optionally only the last :obj:`inputs_embeds` have to be input (see
            :obj:`past_key_values`).
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""
PARALLELIZE_DOCSTRING = r"""
    Uses a device map to distribute attention modules of the model across several devices. If no device map is given,
    it will evenly distribute blocks across all devices.
    Args:
        device_map (:obj:`Dict[int, list]`, optional, defaults to None):
            A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always
            automatically mapped to the first device (for esoteric reasons). That means that the first device should
            have fewer attention modules mapped to it than other devices. For reference, the gpt2 models have the
            following number of attention modules:
                - gpt2: 12
                - gpt2-medium: 24
                - gpt2-large: 36
                - gpt2-xl: 48
    Example::
            # Here is an example of a device map on a machine with 4 GPUs using gpt2-xl, which has a total of 48 attention modules:
            model = GPT2LMHeadModel.from_pretrained('gpt2-xl')
            device_map = {0: [0, 1, 2, 3, 4, 5, 6, 7, 8],
                          1: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
                          2: [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
                          3: [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]}
            model.parallelize(device_map)
"""
DEPARALLELIZE_DOCSTRING = r"""
    Moves the model to cpu from a model parallel state.
    Example::
        # On a 4 GPU machine with gpt2-large:
        model = GPT2LMHeadModel.from_pretrained('gpt2-large')
        device_map = {0: [0, 1, 2, 3, 4, 5, 6, 7],
                    1: [8, 9, 10, 11, 12, 13, 14, 15],
                    2: [16, 17, 18, 19, 20, 21, 22, 23],
                    3: [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]}
        model.parallelize(device_map) # Splits the model across several devices
        model.deparallelize() # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()
"""


@add_start_docstrings(
    "The bare GPT2 Model transformer outputting raw hidden-states without any specific head on top.",
    GPT2_START_DOCSTRING,
)
class GPT2Model(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.init_weights()
        # Model parallel
        self.model_parallel = False
        self.device_map = None

        self.use_layers = None

    def set_layers(self, num_layers):
        assert 1 <= num_layers <= len(self.h)
        if num_layers is not None:
            num_layers -= 1
        self.use_layers = num_layers

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.h), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.h))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        self.wte = self.wte.to(self.first_device)
        self.wpe = self.wpe.to(self.first_device)
        # Load onto devices
        for k, v in self.device_map.items():
            for block in v:
                cuda_device = "cuda:" + str(k)
                self.h[block] = self.h[block].to(cuda_device)
        # ln_f to last
        self.ln_f = self.ln_f.to(self.last_device)

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        self.wte = self.wte.to("cpu")
        self.wpe = self.wpe.to("cpu")
        for index in range(len(self.h)):
            self.h[index] = self.h[index].to("cpu")
        self.ln_f = self.ln_f.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="gpt2",
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
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

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * len(self.h)
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
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
        # position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds  # + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            if self.use_layers is not None and i >= self.use_layers:
                break

            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = layer_past.to(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # checkpointing only works with tuple returns, not with lists
                        return tuple(output for output in module(*inputs, use_cache, output_attentions))

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    layer_past,
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

            hidden_states, present = outputs[:2]
            if use_cache is True:
                presents = presents + (present,)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )



class TrajectoryModel(nn.Module):

    def __init__(self, state_dim, act_dim, max_length=None):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length

    def forward(self, states, actions, rewards, masks=None, attention_mask=None):
        # "masked" tokens or unspecified inputs can be passed in as None
        return None, None, None

    def get_action(self, states, actions, rewards, **kwargs):
        # these will come as tensors on the correct device
        return torch.zeros_like(actions[-1])

class GoalConditionedDecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model 
    """

    def __init__(
            self,
            state_dim,
            goal_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            # Pretrained modules
            GPT_backbone=None,
            embed_timestep=None,
            embed_return=None,
            embed_time_to_goal=None,
            embed_state=None,
            embed_goal=None,
            embed_action=None,
            embed_ln=None,
            predict_state=None,
            predict_goal=None,
            predict_action=None,
            predict_return=None,
            predict_time_to_goal=None,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)
        self.goal_dim = goal_dim

        self.hidden_size = hidden_size

        if GPT_backbone is not None:
            self.transformer = GPT_backbone
            print('[GoalConditionedDecisionTransformer] load GPT backbone from pretrained module')
        else:
            config = transformers.GPT2Config(
                vocab_size=1,  # doesn't matter -- we don't use the vocab
                n_embd=hidden_size,
                **kwargs
            )
            # note: the only difference between this GPT2Model and the default Huggingface version
            # is that the positional embeddings are removed (since we'll add those ourselves)
            self.transformer = GPT2Model(config)
        
        if embed_timestep is not None:
            self.embed_timestep = embed_timestep
        else:
            self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        
        if embed_return is not None:
            self.embed_return = embed_return
        else:
            self.embed_return = torch.nn.Linear(1, hidden_size)
        
        if embed_time_to_goal is not None:
            self.embed_time_to_goal = embed_time_to_goal
        else:
            self.embed_time_to_goal = torch.nn.Linear(1, hidden_size)
        
        if embed_state is not None:
            self.embed_state = embed_state
        else:
            self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        
        if embed_goal is not None:
            self.embed_goal = embed_goal
        else:
            self.embed_goal = torch.nn.Linear(self.goal_dim, hidden_size)
        
        if embed_action is not None:
            self.embed_action = embed_action
        else:
            self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
        
        if embed_ln is not None:
            self.embed_ln = embed_ln
        else:
            self.embed_ln = nn.LayerNorm(hidden_size)

        if predict_state is not None:
            self.predict_state = predict_state
        else:
            self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)

        if predict_goal is not None:
            self.predict_goal = predict_goal
        else:
            self.predict_goal = torch.nn.Linear(hidden_size, self.goal_dim)
        
        if predict_action is not None:
            self.predict_action = predict_action
        else:
            self.predict_action = nn.Sequential(
                *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
            )
        
        if predict_return is not None:
            self.predict_return = predict_return
        else:
            self.predict_return = torch.nn.Linear(hidden_size, 1)
        
        if predict_time_to_goal is not None:
            self.predict_time_to_goal = predict_time_to_goal
        else:
            self.predict_time_to_goal = torch.nn.Linear(hidden_size, 1)
        
    def forward(self, states, goals, actions, rewards, returns_to_go, times_to_goal, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        goal_embeddings = self.embed_goal(goals)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        times_to_goal_embeddings = self.embed_time_to_goal(times_to_goal)
        
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        goal_embeddings = goal_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        times_to_goal_embeddings = times_to_goal_embeddings + time_embeddings

        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, times_to_goal_embeddings, state_embeddings, goal_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 5*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 5*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), times to goal (1), states (2), goals(3) or actions (4); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 5, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        # TODO: try predict action from both decoding of state and goals, even though this operation seems invalid since 
        # casual reasoning of transformer
        return_preds = self.predict_return(x[:,4])  # predict next return given all information
        time_to_goal_preds = self.predict_time_to_goal(x[:,4]) 
        state_preds = self.predict_state(x[:,4])    # predict next state given all information
        goal_preds = self.predict_goal(x[:,4])
        action_preds = self.predict_action(x[:,3])  # predict next action given information without action

        return state_preds, goal_preds, action_preds, return_preds, time_to_goal_preds
    
    def forward_dynamics_prediction(self, states, goals, actions, rewards, returns_to_go, times_to_goal, timesteps, attention_mask):
        batch_size, seq_length = states.shape[0], states.shape[1]

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        goal_embeddings = self.embed_goal(goals)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        times_to_goal_embeddings = self.embed_time_to_goal(times_to_goal)
        
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        goal_embeddings = goal_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        times_to_goal_embeddings = times_to_goal_embeddings + time_embeddings

        stacked_inputs = torch.stack(
            (returns_embeddings, times_to_goal_embeddings, state_embeddings, goal_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 5*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 5*seq_length)

        # Mask return to goal, times to goal, and goal
        forward_dynamics_mask = torch.cat([torch.tensor([0., 0., 1., 0., 1.]) for _ in range(seq_length)]).to(device=states.device, dtype=torch.long)
        stacked_attention_mask = stacked_attention_mask & forward_dynamics_mask

        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']
        x = x.reshape(batch_size, seq_length, 5, self.hidden_size).permute(0, 2, 1, 3)

        state_preds = self.predict_state(x[:,4])    # predict next state given all information (note part of information has been masked)

        return state_preds

    def forward_times_to_goal_prediction(self, states, goals, actions, rewards, returns_to_go, times_to_goal, timesteps, attention_mask):
        batch_size, seq_length = states.shape[0], states.shape[1]

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        goal_embeddings = self.embed_goal(goals)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        times_to_goal_embeddings = self.embed_time_to_goal(times_to_goal)
        
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        goal_embeddings = goal_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        times_to_goal_embeddings = times_to_goal_embeddings + time_embeddings

        stacked_inputs = torch.stack(
            (returns_embeddings, times_to_goal_embeddings, state_embeddings, goal_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 5*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 5*seq_length)

        # Mask return to goal
        times_to_goal_mask = torch.cat([torch.tensor([0., 1., 1., 1., 1.]) for _ in range(seq_length)]).to(device=states.device, dtype=torch.long)
        stacked_attention_mask = stacked_attention_mask & times_to_goal_mask

        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']
        x = x.reshape(batch_size, seq_length, 5, self.hidden_size).permute(0, 2, 1, 3)

        time_to_goal_preds = self.predict_time_to_goal(x[:,4])    # predict times to goal given all information (returns to goal has been masked)

        return time_to_goal_preds

    def forward_masked_sequence_reconstruction(self, states, goals, actions, rewards, returns_to_go, times_to_goal, timesteps, attention_mask):
        batch_size, seq_length = states.shape[0], states.shape[1]

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        goal_embeddings = self.embed_goal(goals)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        times_to_goal_embeddings = self.embed_time_to_goal(times_to_goal)
        
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        goal_embeddings = goal_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        times_to_goal_embeddings = times_to_goal_embeddings + time_embeddings

        stacked_inputs = torch.stack(
            (returns_embeddings, times_to_goal_embeddings, state_embeddings, goal_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 5*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # Mask random
        random_mask = torch.ones((batch_size, seq_length))
        masked_idx = torch.where(torch.rand(batch_size, seq_length) < 0.01)
        random_mask[masked_idx] = 0
        random_mask = random_mask.to(device=states.device, dtype=torch.long)

        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask, attention_mask, attention_mask&random_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 5*seq_length)

        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']
        x = x.reshape(batch_size, seq_length, 5, self.hidden_size).permute(0, 2, 1, 3)

        # return_preds = self.predict_return(x[:,4])  # predict next return given all information
        # time_to_goal_preds = self.predict_time_to_goal(x[:,4]) 
        # state_preds = self.predict_state(x[:,4])    # predict next state given all information
        # goal_preds = self.predict_goal(x[:,4])
        action_recons = self.predict_action(x[:,3])  # predict next action given all information

        return action_recons


    def get_action(self, states, goals, actions, rewards, returns_to_go, times_to_goal, timesteps, **kwargs):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        goals = goals.reshape(1, -1, self.goal_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        times_to_goal = times_to_goal.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            goals = goals[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            times_to_goal = times_to_goal[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            goals = torch.cat(
                [torch.zeros((goals.shape[0], self.max_length-goals.shape[1], self.goal_dim), device=goals.device), goals],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            times_to_goal = torch.cat(
                [torch.zeros((times_to_goal.shape[0], self.max_length-times_to_goal.shape[1], 1), device=times_to_goal.device), times_to_goal],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None
        
        # print('[DEBUG] states shape: {}'.format(states.shape))
        # print('[DEBUG] goals shape: {}'.format(goals.shape))
        # print('[DEBUG] actions shape: {}'.format(actions.shape))
        # print('[DEBUG] returns_to_go shape: {}'.format(returns_to_go.shape))
        # print('[DEBUG] times_to_goal shape: {}'.format(times_to_goal.shape))
        # print('[DEBUG] timesteps shape: {}'.format(timesteps.shape))
        # print('[DEBUG] attention_mask shape: {}'.format(attention_mask.shape))


        _, _, action_preds, _, _ = self.forward(
            states, goals, actions, None, returns_to_go, times_to_goal, timesteps, attention_mask=attention_mask, **kwargs)

        return action_preds[0,-1]


import os
import copy

import numpy as np


from torch.utils.data import Dataset

# Copy from SurRoL simulator
GOAL_DISTANCE = 0.005
SCALE = 1.

class SurRoLDataset(Dataset):
    '''Maximum sequence length is set with 50, 
       demonstrations from scipt will follow this automatically,
       human demonstrations will be downsampled to follow this rule.
       The main concern locates overlong sequence may cause negative influence to performance.'''

    def __init__(self, file_root, seq_max_length, task, with_relabel=True):
        '''Preprocess data from offline dataset'''

        self.T = seq_max_length
        # assert self.T == 100 # Debug
        self.returns_to_goal_scale = self.T
        self.times_to_goal_scale = self.T

        self.data = []
        observations = []
        goals = []

        # Get all files in the file folder
        g = os.walk(file_root)
        for path, _, file_list in g:  
            for file_name in file_list:  
                demo_path = os.path.join(path, file_name)
                if '_' + task + '_' not in demo_path:
                    continue

                print('[SurRoLDataset] valid demo_path: {}'.format(demo_path))
                demo = np.load(demo_path, allow_pickle=True)
                demo_obs, demo_acs = demo['obs'], demo['acs']


                for idx in range(len(demo_obs)):

                    # DEBUG
                    # print('[SurRoLDataset] demo observation length: {}'.format(len(demo_obs[idx])))
                    # print('[SurRoLDataset] demo action length: {}'.format(len(demo_acs[idx])))
                    # END DEBUG

                    seq = {
                        'observations': [],
                        'achieved_goals': [],
                        'goals': [],
                        'actions': [],
                        'rewards': [],
                        'returns_to_goal': [],
                        'times_to_goal': [],
                        'timesteps': [],
                        'success': False,
                        'success_t': -1,
                    }
                    for t in range(len(demo_acs[idx])):
                        # Last observation is remove to match decision transformer's input formulation
                        seq['observations'].append(demo_obs[idx][t]['observation'])
                        observations.append(demo_obs[idx][t]['observation'])
                        seq['achieved_goals'].append(demo_obs[idx][t]['achieved_goal'])
                        seq['goals'].append(demo_obs[idx][t]['desired_goal'])
                        goals.append(demo_obs[idx][t]['desired_goal'])
                        seq['actions'].append(demo_acs[idx][t])
                        seq['rewards'].append(0.0 if np.linalg.norm(demo_obs[idx][t]['achieved_goal'] - demo_obs[idx][t]['desired_goal'], axis=-1) <= GOAL_DISTANCE * SCALE else -1.)
                        if seq['rewards'][-1] == 0.0:
                            seq['success'] = True
                            seq['success_t'] = t
                        seq['timesteps'].append(t)

                    seq['times_to_goal'] = SurRoLDataUtils.calculate_times_to_goal(seq, len(seq['observations']))
                    seq['returns_to_goal'] = SurRoLDataUtils.calculate_returns_to_goal(seq, len(seq['observations']))

                    seq['observations'] = np.array(seq['observations'])
                    seq['achieved_goals'] = np.array(seq['achieved_goals'])
                    seq['goals'] = np.array(seq['goals'])
                    seq['actions'] = np.array(seq['actions'])
                    seq['rewards'] = np.array(seq['rewards'])
                    seq['returns_to_goal'] = np.array(seq['returns_to_goal'])
                    seq['times_to_goal'] = np.array(seq['times_to_goal'])
                    seq['timesteps'] = np.array(seq['timesteps'])

                    self.data.append(seq)
        
        # Hindsight propagate data
        if with_relabel:
            relabeller = HindsightRelabeller(50)
            prop_data = relabeller.step_batch(self.data)
        
            print('[SurRoLDataset] original data sequence number: {}'.format(len(self.data)))
            print('[SurRoLDataset] propagated data sequence number: {}'.format(len(prop_data)))

            self.data += prop_data

        # Calculate mean and std for observation and goal
        observations = np.array(observations)
        goals = np.array(goals)
        
        self.observations_mean = np.mean(observations, axis=0)
        self.observations_std = np.std(observations, axis=0) + 1e-6
        self.goals_mean = np.mean(goals, axis=0)
        self.goals_std = np.std(goals, axis=0) + 1e-6
    
    def pop_normalization_parameters(self):
        return self.observations_mean, self.observations_std, self.goals_mean, self.goals_std


    def __getitem__(self, index):
        observations = self.data[index]['observations']
        goals = self.data[index]['goals']
        actions = self.data[index]['actions']
        rewards = self.data[index]['rewards']
        returns_to_goal = self.data[index]['returns_to_goal']
        times_to_goal = self.data[index]['times_to_goal']
        timesteps = self.data[index]['timesteps']

        # Padding to maximum sequence length
        seq_length = observations.shape[0]
        observations = np.concatenate([np.zeros((self.T - seq_length, observations.shape[-1])), observations], axis=0)
        goals = np.concatenate([np.zeros((self.T - seq_length, goals.shape[-1])), goals], axis=0)
        actions = np.concatenate([np.ones((self.T - seq_length, actions.shape[-1])) * -10., actions], axis=0)
        rewards = np.concatenate([np.ones((self.T - seq_length, 1)) * -1., np.expand_dims(rewards, axis=1)], axis=0)
        returns_to_goal = np.concatenate([np.ones((self.T - seq_length, 1)) * -self.T, np.expand_dims(returns_to_goal, axis=1)], axis=0)
        times_to_goal = np.concatenate([np.ones((self.T - seq_length, 1)) * self.T, np.expand_dims(times_to_goal, axis=1)], axis=0)
        timesteps = np.concatenate([np.zeros((self.T - seq_length)), timesteps], axis=0)
        mask = np.concatenate([np.zeros((self.T - seq_length)), np.ones((seq_length))], axis=0)

        # Normalization for observations and goals
        observations = (observations - self.observations_mean) / self.observations_std
        goals = (goals - self.goals_mean) / self.goals_std
        returns_to_goal = returns_to_goal / self.returns_to_goal_scale
        times_to_goal = times_to_goal / self.times_to_goal_scale

        # print('[DEBUG] observation shape: {}'.format(observations.shape))
        # print('[DEBUG] goals shape: {}'.format(goals.shape))
        # print('[DEBUG] actions shape: {}'.format(actions.shape))
        # print('[DEBUG] rewards shape: {}'.format(rewards.shape))
        # print('[DEBUG] returns_to_goal shape: {}'.format(returns_to_goal.shape))
        # print('[DEBUG] times_to_goal shape: {}'.format(times_to_goal.shape))
        # print('[DEBUG] timesteps shape: {}'.format(timesteps.shape))
        # print('[DEBUG] mask shape: {}'.format(mask.shape))

        return observations, goals, actions, rewards, returns_to_goal, times_to_goal, timesteps, mask

    def __len__(self):
        return len(self.data)
    

class SurRoLDataUtils:
    @staticmethod
    def calculate_times_to_goal(seq, sequence_length):
        if seq['success']:
            times_to_goal = [seq['success_t'] - t for t in range(0, seq['success_t'] + 1)] + \
                            [0 for _ in range(sequence_length - seq['success_t'] - 1)]
        else:
            times_to_goal = [sequence_length for _ in range(sequence_length)]
        
        return times_to_goal

    @staticmethod
    def calculate_returns_to_goal(seq, sequence_length):
        if seq['success']:
            returns_to_goal = [-1.0 * (seq['success_t'] - t) for t in range(0, seq['success_t'] + 1)] + \
                              [0. for _ in range(sequence_length - seq['success_t'] - 1)]
        else:
            returns_to_goal = [-1.0 * sequence_length for _ in range(sequence_length)]
        
        return returns_to_goal

class HindsightRelabeller:
    '''Truncate sequence length to match succes index.'''

    def __init__(self, propagate_num):
        self.propagate_num = propagate_num

    def step(self, sequence):
        seq_length = len(sequence['observations'])
        hindsight_indices = np.random.randint(0, seq_length, size=(self.propagate_num))

        prop_sequences = []
        for idx in hindsight_indices:
            prop_seq = copy.deepcopy(sequence)

            # Change desired goal with achieved goal
            prop_seq['goals'][idx:] = prop_seq['achieved_goals'][idx:]
            prop_seq['success_t'] = idx
            prop_seq['success'] = True

            # Truncate sequence length to remove redundant items
            prop_seq['observations'] = prop_seq['observations'][:idx+1]
            prop_seq['achieved_goals'] = prop_seq['achieved_goals'][:idx+1]
            prop_seq['goals'] = prop_seq['goals'][:idx+1]
            prop_seq['actions'] = prop_seq['actions'][:idx+1]
            prop_seq['rewards'] = prop_seq['rewards'][:idx+1]
            prop_seq['timesteps'] = prop_seq['timesteps'][:idx+1]

            # Recalculate times to goal and returns to goal
            prop_seq['times_to_goal'] = np.array(SurRoLDataUtils.calculate_times_to_goal(prop_seq, idx + 1))
            prop_seq['returns_to_goal'] = np.array(SurRoLDataUtils.calculate_returns_to_goal(prop_seq, idx + 1))

            prop_sequences.append(prop_seq)

            # print('[HindsightRelabeller] truncated sequence length: {}'.format(len(prop_seq['observations'])))

            # print('[HindsightRelabeller] observations shape: {}'.format(prop_seq['observations'].shape))
            # print('[HindsightRelabeller] achieved_goals shape: {}'.format(prop_seq['achieved_goals'].shape))
            # print('[HindsightRelabeller] goals shape: {}'.format(prop_seq['goals'].shape))
            # print('[HindsightRelabeller] actions shape: {}'.format(prop_seq['actions'].shape))
            # print('[HindsightRelabeller] rewards shape: {}'.format(prop_seq['rewards'].shape))
            # print('[HindsightRelabeller] timesteps shape: {}'.format(prop_seq['timesteps'].shape))
            # print('[HindsightRelabeller] times_to_goal shape: {}'.format(prop_seq['times_to_goal'].shape))
            # print('[HindsightRelabeller] returns_to_goal shape: {}'.format(prop_seq['returns_to_goal'].shape))


        return prop_sequences

    def step_batch(self, sequences):
        prop_sequence_batch = []
        for seq in sequences:
            prop_sequence_batch += self.step(seq)
        return prop_sequence_batch
    



if __name__ == '__main__':
    root = '/home/jwfu/goal_conditioned_dt/data/single_demo/'

    surrol_dataset = SurRoLDataset(root, 50, with_relabel=True)
    print(len(surrol_dataset))

    from torch.utils.data import DataLoader

    surrol_dataloader = DataLoader(dataset=surrol_dataset, batch_size=128, shuffle=True)


    for observations, goals, actions, rewards, \
        returns_to_goal, times_to_goal, timesteps, mask in surrol_dataloader:
        print('observations shape: {}'.format(observations.shape))
        print('goals shape: {}'.format(goals.shape))
        print('actions shape: {}'.format(actions.shape))
        print('rewards shape: {}'.format(rewards.shape))
        print('returns_to_goal shape: {}'.format(returns_to_goal.shape))
        print('times_to_goal shape: {}'.format(times_to_goal.shape))
        print('timesteps shape: {}'.format(timesteps.shape))
        print('mask shape: {}'.format(mask.shape))
