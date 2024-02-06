""" MetaTree Model Class"""
import math
from typing import List, Optional, Tuple, Union, Any

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from transformers.models.llama.configuration_llama import LlamaConfig

from metatree.modeling_outputs import CausalMetaTreeOutputWithPast

import copy
from torchvision.ops import MLP
from metatree.generate_util import TreeGenerationMixin
import einops

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"

EPS = 1e-5

# Generation Helpers

def entropy(labels):
    # Calculate the frequency of each class
    vals, counts = labels.unique(return_counts=True)
    probs = counts.float() / labels.size(0)
    log_probs = torch.log2(probs)
    # Calculate entropy
    return -torch.sum(probs * log_probs)

def entropy_construction(inputs, labels):
    # inputs: [seq_len, num_features]
    # labels: [seq_len] (have to be class integers)
    num_features = inputs.shape[1]
    entropy_matrix = torch.zeros_like(inputs)
    original_entropy = entropy(labels)

    for feature_index in range(num_features):
        # Unique values and their indices
        unique_values, indices = torch.unique(inputs[:, feature_index], sorted=True, return_inverse=True)
        
        # Iterate over potential splits
        for value in unique_values:
            left_mask = inputs[:, feature_index] <= value
            right_mask = ~left_mask

            left_entropy = entropy(labels[left_mask])
            right_entropy = entropy(labels[right_mask])

            # Weighted average entropy
            total_entropy = (torch.sum(left_mask) * left_entropy + torch.sum(right_mask) * right_entropy) / len(labels)

            # assign the entropy to the feature
            entropy_matrix[:, feature_index] = torch.where(inputs[:, feature_index] == value, total_entropy, entropy_matrix[:, feature_index])
    entropy_matrix = (original_entropy - entropy_matrix) / (original_entropy+EPS) # convert to information gain
    return entropy_matrix



# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), float("-inf"))


# Make the input mask communitive
def _make_communitive_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    this mask should be 1 denoting input bits, zero denoteing other bits
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return (self.weight * hidden_states).to(input_dtype)


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(x.dtype), persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(x.dtype), persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            )

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.self_row_attn = LlamaAttention(config=config)
        self.self_column_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        row_attention_mask: Optional[torch.Tensor] = None,
        column_attention_mask: Optional[torch.Tensor] = None,
        row_position_ids: Optional[torch.LongTensor] = None,
        column_position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        bs, hidden_len, _ = hidden_states.size()
        hidden_seq_len = int(hidden_len / self.config.n_feature)

        hidden_states = self.input_layernorm(hidden_states)

        # Now Hidden States are in the shape of [bs, m * n, d] -> Factor shape first
        hidden_states = hidden_states.view(bs, hidden_seq_len, self.config.n_feature, self.config.hidden_size)
        hidden_states_row = hidden_states.transpose(1,2).contiguous().view(bs * self.config.n_feature, hidden_seq_len,  self.config.hidden_size)
        hidden_states_column = hidden_states.view(bs * hidden_seq_len, self.config.n_feature, self.config.hidden_size)

        # Self Attention
        hidden_states_row, self_attn_weights, present_key_value = self.self_row_attn(
            hidden_states=hidden_states_row,
            attention_mask=row_attention_mask,
            position_ids=row_position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        hidden_states_column, self_attn_weights, present_key_value = self.self_column_attn(
            hidden_states=hidden_states_column,
            attention_mask=column_attention_mask,
            position_ids=column_position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        hidden_states_row = hidden_states_row.view(bs, self.config.n_feature, hidden_seq_len, self.config.hidden_size).transpose(1, 2).contiguous().view(bs, hidden_seq_len * self.config.n_feature, self.config.hidden_size)
        hidden_states_column = hidden_states_column.view(bs, hidden_seq_len, self.config.n_feature, self.config.hidden_size).view(bs, hidden_seq_len * self.config.n_feature, self.config.hidden_size)
        
        hidden_states = residual + hidden_states_row + hidden_states_column

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlamaModel):
            module.gradient_checkpointing = value


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.n_class = config.n_class
        self.n_feature = config.n_feature
        self.depth = config.depth

        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = config.gradient_checkpointing
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.class_emb

    def set_input_embeddings(self, value):
        self.class_emb = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        
        if input_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_input_mask = _make_communitive_mask(input_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_input_mask if combined_attention_mask is None else - expanded_input_mask * combined_attention_mask
            )

        return combined_attention_mask

    def _prepare_attention_mask(self, attention_mask, input_shape, inputs_embeds):
        # create attention mask, input will be communitive, and masked elements will be ignored
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def _attn_mask_2_row_col_mask(self, attention_mask):
        # Extract the Row & Column Mask from the Sequence Attention & Input Mask
        bs, seq_len = attention_mask.shape
        seq_len = seq_len // self.config.n_feature
        n_feature = self.config.n_feature
        attention_mask = attention_mask.reshape(bs, seq_len, n_feature)
        column_attn_mask = attention_mask.view(bs * seq_len, n_feature)
        #column_attn_mask = torch.ones(bs * seq_len, n_feature, dtype=torch.bool, device=attention_mask.device)
        row_attn_mask = attention_mask.transpose(1,2).contiguous().view(bs * n_feature, seq_len)
        return row_attn_mask, column_attn_mask

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_x: Optional[torch.FloatTensor] = None,
        input_y: Optional[torch.LongTensor] = None,
        state_vec: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length, _ = inputs_embeds.shape

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        row_attention_mask, column_attention_mask = self._attn_mask_2_row_col_mask(attention_mask)
        seq_length = seq_length // self.config.n_feature

        # Use no ROPE embeddings
        row_position_ids = None
        column_position_ids = None

        row_attention_mask = self._prepare_attention_mask(row_attention_mask, (batch_size * self.config.n_feature, seq_length), inputs_embeds)
        column_attention_mask = self._prepare_attention_mask(column_attention_mask, (batch_size * seq_length, self.config.n_feature), inputs_embeds)

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    None,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    row_attention_mask=row_attention_mask,
                    column_attention_mask=column_attention_mask,
                    row_position_ids=row_position_ids,
                    column_position_ids=column_position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForMetaTree(TreeGenerationMixin, LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.main_input_name = "input_embeds"

        self.model = LlamaModel(config)
                
        self.class_proj = nn.Linear(config.n_class, config.hidden_size)
        self.linear_proj = nn.Linear(1, config.hidden_size)

        self.emb_bias = nn.Parameter(torch.zeros(config.n_feature, config.hidden_size))
        self.pos_bias = nn.Parameter(torch.zeros(config.max_position_embeddings, config.hidden_size))

        self.x_mlp = LlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )

        # now we get the best tree of a given depth
        self.depth = config.depth
        self.normalize = config.normalize
        self.config = config
        self.classifier = nn.Linear(config.hidden_size, 1)
        
        # Initialize weights and apply final processing
        self.post_init()

    def _find_smallest_element(self, tensor):
        # NO GRADIENTS SHOULD PASS THROUGH THIS FUNCTION
        # Create a boolean mask of positive elements
        # target_tsr = tensor.clone()
        # Find the smallest positive element
        smallest_value, smallest_index = torch.min(tensor, dim=-1)
        return smallest_value, smallest_index

    def _onehot_argmax(self, input_vector, dim):
        # Find the argmax indices along the specified dimension
        argmax_indices = torch.argmax(input_vector, dim=dim)
        # Create a one-hot tensor based on the argmax indices
        onehot = torch.zeros_like(input_vector)
        onehot.scatter_(dim=dim, index=argmax_indices.unsqueeze(dim), value=1)
        return onehot

    def _construct_target_matrix(self, truth_split, truth_mask):
        sigma = self.config.sigma
        # important to have this normalization
        normalized_truth_split = truth_split / (truth_split.max(dim=-1, keepdim=True)[0] + EPS)
        truth_split_gaussian = torch.exp(-torch.pow(normalized_truth_split.float(), 2) / (2 * sigma ** 2))
        oh_split = truth_split_gaussian
        oh_mask = F.one_hot(truth_mask, num_classes=self.config.n_feature).float()
        target_matrix = torch.einsum('bl,bk->blk', oh_split, oh_mask).float()
        return target_matrix

    def _find_parent(self, idx):
        return (idx-1) // 2

    def _find_path(self, idx):
        # idx is the index of the leaf node
        # return the path from the leaf node to the root, excluding the root
        if idx == 0:
            return []
        return [idx] + self._find_path((idx - 1)// 2)

    def _get_decision(self, decision_lst, path):
        final_decision = torch.ones_like(decision_lst[0])

        for idx in path:
            if idx % 2 == 1: #Last split = TRUE
                final_decision = final_decision * decision_lst[self._find_parent(idx)]
            else: #Last split = FALSE
                final_decision = final_decision * (1 - decision_lst[self._find_parent(idx)])
        return final_decision
    
    def get_input_embeddings(self):
        return self.linear_proj

    def set_input_embeddings(self, value):
        self.linear_proj = value

    def get_output_embeddings(self):
        return self.metatree_head

    def set_output_embeddings(self, new_embeddings):
        self.metatree_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def _get_input_embeds(self, input_x, input_y, status):
        # get input embeddings 
        # input_x: [bsz, seq_len, n_feature]
        # input_y: [bsz, seq_len, n_class] <- one-hot
        # state_vec: [bsz, seq_len, 2*n_feature] <- split threshold + split dimension
        # output: [bsz, seq_len, hidden_size]
        device = input_x.device
        cpu_device = torch.device("cpu")
        # TODO: Normalization In-Place
        status_mask = status.unsqueeze(-1)
        input_x_ = input_x.clone()
        seq_len = input_x.shape[1]

        if self.normalize:
            input_x_mean = (input_x_ * status_mask).sum(dim=1, keepdim=True)/ (status_mask.sum(dim=1, keepdim=True) + EPS)
            input_x_std = torch.sqrt(((input_x_ - input_x_mean) * status_mask).pow(2).sum(dim=1, keepdim=True)/ (status_mask.sum(dim=1, keepdim=True) + EPS))
            input_x_ = (input_x_ - input_x_mean) * status_mask / (input_x_std + EPS)
            #input_x_ =  (input_x_ - input_x_.mean(dim=1, keepdim=True)) / (input_x_.std(dim=1, keepdim=True) + EPS)

        # First Pad the unused dimension
        input_y_pad = F.pad(input_y, (0, self.config.n_class - input_y.shape[-1]), mode='constant', value=0)
        input_x_pad = F.pad(input_x_, (0, self.config.n_feature - input_x.shape[-1]), mode='constant', value=0)
        
        input_x_embed = self.linear_proj(input_x_pad.unsqueeze(-1)) + self.emb_bias + self.pos_bias[:seq_len,:].unsqueeze(1) # bs, seq, n_feature, hidden_size 
        input_x_embed = input_x_embed + self.class_proj(input_y_pad).unsqueeze(2)
        input_x_embed = self.x_mlp(input_x_embed)

        return input_x_embed

    def step(
        self,
        input_x: Optional[torch.FloatTensor] = None,
        input_y: Optional[torch.LongTensor] = None,
        status: Optional[torch.FloatTensor] = None, # n_data x time_step
        split_dimension: Optional[torch.FloatTensor] = None, # n_feature x time_step
        split_threshold: Optional[torch.FloatTensor] = None, # n_data x time_step
        reward: Optional[torch.FloatTensor] = None, # bs, 1
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        generate: Optional[bool] = None,
        deterministic: Optional[bool] = True,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        assert input_x is not None or inputs_embeds is not None, "You have to specify either input ids or input embeddings"
        assert input_x is None or inputs_embeds is None, "You cannot specify both input_ids and inputs_embeds at the same time"

        loss = 0
        decision_loss = 0
        log_prob = 0
        bs, input_len, _ = input_x.shape
        dim_len = self.config.n_feature

        if self.training:
            # Shuffle the input over the feature dimension
            idx = torch.randperm(input_x.shape[2]).to(input_x.device)
            input_x = input_x[:,:,idx]
            split_dimension = idx[split_dimension.long()].to(split_dimension.device)
        

        if input_x is not None:
            inputs_embeds = self._get_input_embeds(input_x, input_y, status)
            inputs_embeds = inputs_embeds.view(bs, -1, self.config.hidden_size)

        # Masking for the input
        feature_sum = input_x.abs().sum(dim=1)
        feature_mask = torch.ones_like(feature_sum).masked_fill(feature_sum == 0, 0)
        feature_mask = feature_mask.unsqueeze(1).expand(bs, input_len, dim_len) # Shape = [bs, n_feature] -> [bs, seq_len, n_feature]

        status = status.float()
        status_mask = status.unsqueeze(-1).expand(bs, input_len, dim_len) # Shape = [bs, seq_len] -> [bs, seq_len, n_feature]
        status_mask = status_mask * feature_mask # No Feature Mask
        status_mask = status_mask.reshape(bs, input_len * dim_len)
        seq_mask = torch.zeros_like(status).masked_fill(status==0, float('-inf'))
        output_mask = torch.zeros_like(status_mask).masked_fill(status_mask==0, float('-inf'))
        label_mask = torch.ones_like(status_mask).masked_fill(status_mask==0, 0)
        input_mask = status_mask
        attention_mask = status_mask 
        
        # Run the model
        outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                input_mask=input_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        hidden_states = outputs[0]

        output_states = self.classifier(hidden_states).squeeze(-1)

        # Now we need to do one step of the metatree
        tentative_y = input_y
        tentative_x = input_x
        BCE_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        CE_loss = CrossEntropyLoss(weight=torch.tensor([1.0, 1.0 * input_len * dim_len], device=input_x.device))
        KL_loss = torch.nn.KLDivLoss(reduction='batchmean')
        MSE_loss = torch.nn.MSELoss(reduction='mean')
        L1_loss = torch.nn.L1Loss(reduction='mean')

        if not generate:
            target_matrix = self._construct_target_matrix(split_threshold, split_dimension).view(bs, input_len * dim_len) * label_mask
        # Get the Split Information
        decision_matrix = torch.sigmoid(output_states + output_mask)
        #decision_matrix = decision_matrix 
        decision_matrix = decision_matrix.view(bs, input_len, dim_len)

        # Max Out Prediction
        if deterministic:
            final_decision = torch.zeros_like(decision_matrix).view(bs, -1)
            final_decision.scatter_(1, decision_matrix.view(bs,-1).argmax(dim=1, keepdim=True), 1)
            final_decision = final_decision.view(bs, input_len, dim_len)
        else:
            # TOP-K Sampling (not used in paper evaluation, but maybe useful)
            temperature = 0.5
            nucleus_p = 0.3
            gamma = 0.8
            k = 2

            sampling_decision_matrix = torch.softmax(output_states + output_mask / temperature, dim=-1) 
            if status_mask.sum() <= 0:
                # If all the datapoints are not related, then we just sample from the distribution
                sampling_decision_matrix = torch.nan_to_num(sampling_decision_matrix, nan=1/decision_matrix.shape[-1])
                sampling_decision_matrix += 1/decision_matrix.shape[-1]

            top_k_values, top_k_indices = torch.topk(sampling_decision_matrix, k)
    
            # Zero out the probabilities of all tokens except the top k
            topk_mask = torch.zeros_like(sampling_decision_matrix, dtype=torch.bool)
            topk_mask.scatter_(1, top_k_indices, True)
            filtered_probs = torch.where(topk_mask, sampling_decision_matrix, torch.full_like(sampling_decision_matrix, 0))

            final_decision = sampling_decision.sample()
            final_decision = final_decision.view(bs, input_len, dim_len)
        
        
        metatree_split = torch.sum(final_decision, dim=2) # n_data x n_feature
        metatree_mask = torch.sum(final_decision, dim=1) # n_data x n_data

        assert metatree_mask.max() == 1, "Multiple Splitting?"
        
        split_idx = metatree_mask.argmax(dim=-1)
        tentative_x = torch.gather((input_x + 1e-6), dim=-1, index=split_idx.view(bs,1,1).expand(bs,input_len,1)).squeeze(-1)
        tentative_split = torch.einsum('bl,bl->b', tentative_x, metatree_split).unsqueeze(-1)

        if not generate:
            truth_split, truth_mask = split_threshold, split_dimension
            loss_mask = torch.log(status_mask + EPS)
            assert truth_split.shape[0] == metatree_split.shape[0], "Batch size mismatch"

            loss = loss + BCE_loss(output_states + loss_mask, target_matrix.float())
            decision = (tentative_x < tentative_split).float() * status
            assert decision.max() <= 1 and decision.min() >= 0, f"Decision Irregular, {decision.max()}, {decision.min()}"
            assert decision.max() >= 1 or decision.min() <= 0, f"Decision out of bound, {decision.max()}, {decision.min()}"
        else:
            # Generate the next decision with no loss
            decision = (tentative_x < tentative_split).float() * status
            assert decision.max() <= 1 and decision.min() >= 0, f"Decision Irregular, {decision.max()}, {decision.min()}"
            assert decision.max() >= 1 or decision.min() <= 0, f"Decision out of bound, {decision.max()}, {decision.min()}"
                
        return decision, loss, metatree_mask, metatree_split, log_prob, tentative_split, decision_matrix, outputs

    def forward(
        self,
        input_x: Optional[torch.FloatTensor] = None,
        input_y: Optional[torch.LongTensor] = None,
        status: Optional[torch.FloatTensor] = None, # n_data x time_step
        split_dimension: Optional[torch.FloatTensor] = None, # n_feature x time_step
        split_threshold: Optional[torch.FloatTensor] = None, # n_data x time_step
        reward: Optional[torch.FloatTensor] = None,
        context: Optional[Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        generate: Optional[bool] = None,
        accelerator: Optional[Any] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        assert input_x is not None or inputs_embeds is not None, "You have to specify either input ids or input embeddings"
        assert input_x is None or inputs_embeds is None, "You cannot specify both input_ids and inputs_embeds at the same time"
        
        output_dim = input_x.shape[-1] if input_x is not None else inputs_embeds.shape[-1]

        #We start the forward pass
        loss = 0
        decision_loss = 0
        bs, input_len, _ = input_x.shape
        decision_lst = []
        log_probs = []

        NLL_loss = torch.nn.NLLLoss()
        MSE_loss = MSELoss()
        CE_loss = CrossEntropyLoss()

        decision = None
        status_lst, metatree_masks, metatree_splits, tentative_splits = [], [], [], []
        generate_status_lst = [torch.ones((bs, input_len), device=input_x.device, dtype=input_x.dtype)]
        status = self.get_ground_truth_status(input_x, input_y, split_threshold, split_dimension)
        # memory replay backprop 
        for ctr_split in range(split_threshold.shape[1]):
            # Get the Selected Splits
            t_threshold, t_dimension = split_threshold[:,ctr_split], split_dimension[:,ctr_split]
            if not generate:
                t_status = status[ctr_split]
                #t_status = status[:,ctr_split]
            else:
                t_status = generate_status_lst.pop(0)
                #print(t_status.sum(dim=-1)) # For debug
            status_lst.append(t_status)

            t_x = input_x
            t_y = input_y

            decision, t_loss, metatree_mask, metatree_split, log_prob, tentative_split, decision_matrix, outputs = self.step(input_x=t_x, input_y=t_y, status=t_status, split_dimension=t_dimension, split_threshold=t_threshold, generate=generate, reward=reward, output_hidden_states=output_hidden_states,)

            metatree_masks.append(metatree_mask.detach())
            metatree_splits.append(metatree_split.detach())
            tentative_splits.append(tentative_split.detach())

            if generate:
                generate_status_lst.append(t_status * decision)
                generate_status_lst.append(t_status * (1-decision))
            
            # THIS Piece works when accelerator.no_sync() is used
            if self.config.backward_window > 0:
                if (ctr_split+1) % self.config.backward_window == 0 and ctr_split < split_threshold.shape[1] - 1:
                    loss = loss + t_loss
                    if self.training:
                        accelerator.backward(loss)
                        loss = loss.detach()
                    decision_lst.append(decision.detach())
                else:
                    decision_lst.append(decision.detach())
                    loss = loss + t_loss
            else:
                loss = loss + t_loss
                decision_lst.append(decision)
            
            if output_hidden_states:
                # Output the Hidden States of the First Split for Exploration
                return CausalMetaTreeOutputWithPast(
                loss=loss,
                status=status_lst,
                log_probs=log_probs,
                metatree_thresolds=metatree_splits,
                metatree_dimensions=metatree_masks,
                tentative_splits=tentative_splits,
                logits=None,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        y_pred = torch.zeros_like(input_y).float()
        # iterate over leaf nodes
        for tree_ctr in range(2**self.depth-1, 2**(self.depth+1)-1):
            path = self._find_path(tree_ctr)
            decision = self._get_decision(decision_lst, path).unsqueeze(-1) # BS, Seq, 1
            y_pred += decision * ((input_y * decision).sum(dim=1, keepdim=True) / (decision.sum(dim=1, keepdim=True) + EPS)) # BS, Seq, Nclass

        if generate:
            decision_loss = NLL_loss((y_pred.view(-1, input_y.shape[-1]) + EPS).log(), input_y.argmax(dim=-1).view(-1))
            loss += decision_loss

        return CausalMetaTreeOutputWithPast(
            loss=loss,
            status=status_lst,
            log_probs=log_probs,
            metatree_thresolds=metatree_splits,
            metatree_dimensions=metatree_masks,
            tentative_splits=tentative_splits,
            logits=y_pred,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
    
    def generate_decision_tree(
        self, 
        input_x: torch.FloatTensor, # shape = [seq_len, n_feature]
        input_y: torch.LongTensor, # shape = [seq_len, n_class]
        depth: int = 2,
        deterministic: bool = True,
        **kwargs,
    ):
        ## Generate a decision tree from the model
        ## Input: X, Y
        ## Output: Decision Tree, but not the decision path
        ## The decsion tree will be evaluated later on entire test set

        if len(input_x.shape) == 2:
            input_x = input_x.unsqueeze(0)
            input_y = input_y.unsqueeze(0)
            # map -> [1, seq_len, n_feature]
        assert len(input_x.shape) == 3, "Input X should be of shape [bs, seq_len, n_feature]"
        bs, input_len, dim_len = input_x.shape
        decision_lst = []
        log_probs = []

        loss = 0
        decision = None
        status_lst, metatree_masks, metatree_splits, tentative_splits = [], [], [], []
        generate_status_lst = [torch.ones((bs, input_len), device=input_x.device, dtype=input_x.dtype)]

        for ctr_split in range(2**depth-1):
            t_status = generate_status_lst.pop(0)
            status_lst.append(t_status)
            t_x = input_x
            t_y = input_y

            decision, t_loss, metatree_mask, metatree_split, log_prob, tentative_split, decision_matrix, _ = self.step(input_x=t_x, input_y=t_y, status=t_status, generate=True, deterministic=deterministic)
            # Finding the middle point

            # batch size has to be one for this:
            #import pdb; pdb.set_trace()
            tt_x = input_x * t_status.unsqueeze(-1).expand(bs, input_len, dim_len)
            tt_x = torch.gather(input_x + 1e-6, dim=-1, index=metatree_mask.argmax(-1).view(bs,1,1).expand(bs,input_len,1))
            delta = tt_x - tentative_split.unsqueeze(-1)
            if delta.max() <= 0:
                pass
            else:
                add_on_idx = torch.argmin(delta[delta > 0], dim=-1)
                add_on_val = tt_x[delta > 0][add_on_idx]
                tentative_split = (tentative_split + add_on_val) / 2


            metatree_masks.append(metatree_mask.detach())
            metatree_splits.append(metatree_split.detach())
            tentative_splits.append(tentative_split.detach())

            generate_status_lst.append(t_status * decision)
            generate_status_lst.append(t_status * (1-decision))
            
            loss = loss + t_loss
            decision_lst.append(decision)
                
        y_pred = torch.zeros_like(input_y).float()
        # iterate over leaf nodes
        for tree_ctr in range(2**self.depth-1, 2**(self.depth+1)-1):
            path = self._find_path(tree_ctr)
            decision = self._get_decision(decision_lst, path).unsqueeze(-1) # BS, Seq, 1
            y_pred += decision * ((input_y * decision).sum(dim=1, keepdim=True) / (decision.sum(dim=1, keepdim=True) + EPS)) # BS, Seq, Nclass


        return CausalMetaTreeOutputWithPast(
            loss=loss,
            status=status_lst,
            log_probs=log_probs,
            metatree_thresolds=metatree_splits,
            metatree_dimensions=metatree_masks,
            tentative_splits=tentative_splits,
            logits=y_pred,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def get_ground_truth_status(self, input_x, input_y, split_threshold, split_dimension):
        bs, input_len, _ = input_x.shape
        decision_lst= []
        tentative_y = input_y
        status_out_lst = []
        status_lst = [torch.ones_like(input_x[:,:,0])]
        for ctr_split in range(split_threshold.shape[1]):
            status_cur = status_lst.pop(0)
            status_out_lst.append(status_cur)
            truth_split, truth_mask = split_threshold[:,ctr_split], split_dimension[:,ctr_split]
            truth_split_val = truth_split.min(dim=-1, keepdim=True)[0]
            truth_split = (truth_split == truth_split_val).float()
            truth_split = truth_split / (truth_split.sum(dim=-1, keepdim=True) + 1e-5)
            # Get the Selected Splits
            split_idx = truth_mask
            tentative_x = torch.gather((input_x + 1e-5), dim=-1, index=split_idx.view(bs,1,1).expand(bs,input_len,1)).squeeze(-1)
            tentative_split = torch.einsum('bl,bl->b', tentative_x, truth_split).unsqueeze(-1)
            decision = (tentative_x < tentative_split).float() * status_cur
            assert decision.max() <= 1 and decision.min() >= 0, f"Decision Irregular, {decision.max()}, {decision.min()}"
            assert decision.max() >= 1 or decision.min() <= 0, f"Decision out of bound, {decision.max()}, {decision.min()}"
            decision_lst.append(decision)
            status_lst.append(status_cur * decision)
            status_lst.append(status_cur * (1-decision))
        return status_out_lst
    
    def evaluate_decision_tree(self, input_x, input_y, split_threshold, split_dimension):
        bs, input_len, _ = input_x.shape
        decision_lst= []
        tentative_y = input_y
        status_lst = [torch.ones_like(input_x[:,:,0])]
        for ctr_split in range(split_threshold.shape[1]):
            status_cur = status_lst.pop(0)
            truth_split, truth_mask = split_threshold[:,ctr_split], split_dimension[:,ctr_split]
            truth_split_val = truth_split.min(dim=-1, keepdim=True)[0]
            truth_split = (truth_split == truth_split_val).float()
            truth_split = truth_split / (truth_split.sum(dim=-1, keepdim=True) + 1e-5)
            # Get the Selected Splits
            split_idx = truth_mask
            tentative_x = torch.gather((input_x + 1e-5), dim=-1, index=split_idx.view(bs,1,1).expand(bs,input_len,1)).squeeze(-1)
            tentative_split = torch.einsum('bl,bl->b', tentative_x, truth_split).unsqueeze(-1)
            decision = (tentative_x < tentative_split).float() * status_cur
            assert decision.max() <= 1 and decision.min() >= 0, f"Decision Irregular, {decision.max()}, {decision.min()}"
            assert decision.max() >= 1 or decision.min() <= 0, f"Decision out of bound, {decision.max()}, {decision.min()}"
            decision_lst.append(decision)
            status_lst.append(status_cur * decision)
            status_lst.append(status_cur * (1-decision))
        y_pred = torch.zeros_like(tentative_y).float()

        # iterate over leaf nodes
        for tree_ctr in range(2**self.depth-1, 2**(self.depth+1)-1):
            path = self._find_path(tree_ctr)
            decision = self._get_decision(decision_lst, path).unsqueeze(-1) # BS, Seq, 1
            y_pred += decision * ((tentative_y * decision).sum(dim=1, keepdim=True) / (decision.sum(dim=1, keepdim=True) + EPS)) # BS, Seq, Nclass
        return y_pred
    
    def evaluate_decision_tree_with_generalization(self, input_x, input_y, split_threshold, split_dimension, gen_x, gen_y):
        bs, input_len, _ = input_x.shape
        decision_lst= []
        tentative_y = input_y
        status_lst = [torch.ones_like(gen_x[:,0])]
        for ctr_split in range(split_threshold.shape[1]):
            status_cur = status_lst.pop(0)
            truth_split, truth_mask = split_threshold[:,ctr_split], split_dimension[:,ctr_split]
            truth_split_val = truth_split.min(dim=-1, keepdim=True)[0]
            truth_split = (truth_split == truth_split_val).float()
            truth_split = truth_split / (truth_split.sum(dim=-1, keepdim=True) + 1e-5)
            # Get the Selected Splits
            split_idx = truth_mask
            tentative_x = torch.gather((input_x + 1e-5), dim=-1, index=split_idx.view(bs,1,1).expand(bs,input_len,1)).squeeze(-1)
            tentative_split = torch.einsum('bl,bl->b', tentative_x, truth_split).unsqueeze(-1)
            decision = (tentative_x < tentative_split).float() * status_cur
            #decision[0,:] == status[0,ctr_split+1]
            assert decision.max() <= 1 and decision.min() >= 0, f"Decision Irregular, {decision.max()}, {decision.min()}"
            assert decision.max() >= 1 or decision.min() <= 0, f"Decision out of bound, {decision.max()}, {decision.min()}"
            decision_lst.append(decision)
            status_lst.append(status_cur * decision)
            status_lst.append(status_cur * (1-decision))
        y_pred = torch.zeros_like(tentative_y).float()

        # iterate over leaf nodes
        for tree_ctr in range(2**self.depth-1, 2**(self.depth+1)-1):
            path = self._find_path(tree_ctr)
            decision = self._get_decision(decision_lst, path).unsqueeze(-1) # BS, Seq, 1
            y_pred += decision * ((tentative_y * decision).sum(dim=1, keepdim=True) / (decision.sum(dim=1, keepdim=True) + EPS)) # BS, Seq, Nclass
        #for _ in range(2): print(torch.mean((y_pred[_,:].argmax(-1) == tentative_y[_,:].argmax(-1)).float()))
        return y_pred


    def prepare_inputs_for_generation(
        self, input_x=None, input_y=None, split_threshold=None, split_dimension=None, input_ids=None, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):  
        #TODO: Cache Management Later
        model_inputs=(
            {
                "input_x": input_x,
                "input_y": input_y,
                "split_threshold": split_threshold,
                "split_dimension": split_dimension,
                "inputs_embeds": inputs_embeds,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past