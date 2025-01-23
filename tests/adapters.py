#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import IO, BinaryIO, Iterable, Optional, Type

import numpy.typing as npt
import torch
import torch.nn.functional as F
import regex as re
from collections import defaultdict, Counter
from .model import RMSNorm, GELU, FFN


def run_positionwise_feedforward(
    d_model: int,
    d_ff: int,
    weights: dict[str, torch.FloatTensor],
    in_features: torch.FloatTensor,
) -> torch.FloatTensor:
    """Given the weights of a position-wise feedforward network, return
    the output of your implementation with these weights.

    Args:
        d_model: int
            Dimensionality of the feedforward input and output.
        d_ff: int
            Dimensionality of the feedforward network's inner layer.
        weights: dict[str, torch.FloatTensor]
            State dict of our reference implementation.
            The keys of this dictionary are `w1.weight` and `w2.weight`.
            `w1` is the first linear transformation, and `w2` is the second
            linear transformation (eq. 2 of Vaswani et al., 2017).
            `w1.weight` is of shape (d_ff, d_model).
            `w2.weight` is of shape (d_model, d_ff).
    )
        in_features: torch.FloatTensor
            Tensor to run your implementation on.

    Returns:
        torch.FloatTensor with the output of running your position-wise feedforward network
        with the provided `weights` on the provided `in_features`.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # my_ffn.load_state_dict(weights)
    # You can also manually assign the weights
    # my_ffn.w1.weight.data = weights["w1.weight"]
    # my_ffn.w2.weight.data = weights["w2.weight"]

    ffn = FFN(d_model=d_model, d_fnn=d_ff)
    ffn.load_state_dict(weights)

    output = ffn(in_features)

    return output


def run_scaled_dot_product_attention(
    K: torch.FloatTensor,
    Q: torch.FloatTensor,
    V: torch.FloatTensor,
    mask: Optional[torch.BoolTensor] = None,
    pdrop: Optional[float] = None,
) -> torch.FloatTensor:
    """Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        K: torch.FloatTensor
            Tensor with attention keys. Shape is
            (batch_size, ..., seq_len, key_dimension), where
            "..." is optional and represents any number of other
            batch dimensions (e.g., num_heads).
        Q: torch.FloatTensor
            Tensor with attention queries. Shape is
            (batch_size, ..., seq_len, key_dimension), where
            "..." is optional and represents any number of other
            batch dimensions (e.g., num_heads).
        V: torch.FloatTensor
            Tensor with attention values. Shape is
            (batch_size, ..., seq_len, value_dimension), where
            "..." is optional and represents any number of other
            batch dimensions (e.g., num_heads).
        mask: Optional[torch.BoolTensor]
            An (optional) mask of shape (seq_len, seq_len).
            Attention scores for positions with a mask value of `True` should
            be masked out, i.e., not affect the softmaxed attention probabilities.
        pdrop: Optional[float], default is None.
            If given, drop-out the attention probabilities (the softmax-normalized
            attention scores) with this rate.

    Returns:
        torch.FloatTensor of shape (batch_size, ..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """
    tmp_Q = torch.unsqueeze(Q, dim=1) if len(Q.shape) == 3 else Q
    tmp_K = torch.unsqueeze(K, dim=1) if len(K.shape) == 3 else K
    tmp_V = torch.unsqueeze(V, dim=1) if len(V.shape) == 3 else V
    
    d_k = tmp_Q.shape[-1]
    qk = tmp_Q @ tmp_K.transpose(2,3) / (d_k ** 0.5)

    if mask is not None:
        mask_values = torch.where(mask, float('-inf'), 0.0)
        qk = qk + mask_values

    softmax_dk = run_softmax(qk, dim=-1)
    output = softmax_dk @ tmp_V

    output = torch.squeeze(output) if len(Q.shape) == 3 else output

    if pdrop is not None:
        output = F.dropout(output, p=pdrop, training=True)

    return output


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    attn_pdrop: float,
    weights: dict[str, torch.FloatTensor],
    in_features: torch.FloatTensor,
) -> torch.FloatTensor:
    """Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model: int
            Dimensionality of the feedforward input and output.
        num_heads: int
            Number of heads to use in multi-headed attention.
        attn_pdrop: float
            Drop-out the attention probabilities (the softmax-normalized
            attention scores) with this rate.
        weights: dict[str, torch.FloatTensor]
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `q_heads.{N}.weight`, `q_heads.{N}.weight`:
                Weights for the query projection heads.
                N is an integer from 0 to `num_heads - 1`.
                Shape of each tensor is (d_key, d_model).
            - `k_heads.{N}.weight`, `k_heads.{N}.weight`:
                Weights for the key projection heads.
                N is an integer from 0 to `num_heads - 1`.
                Shape of each tensor is (d_key, d_model).
            - `v_heads.{N}.weight`, `v_heads.{N}.weight`:
                Weights for the value projection heads.
                N is an integer from 0 to `num_heads - 1`.
                Shape of each tensor is (d_value, d_model).
            - `output_proj.weight`:
                Weight of the output projection
                (W^{O} in the original Transformer paper)
                Shape of (d_model, d_value * num_heads).
        in_features: torch.FloatTensor
            Tensor to run your implementation on.

    Returns:
        torch.FloatTensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    concated_q_weight = torch.concat([weights[f"q_heads.{i}.weight"] for i in range(num_heads)], dim=0) # (head * d_key, d_model)
    concated_k_weight = torch.concat([weights[f"k_heads.{i}.weight"] for i in range(num_heads)], dim=0)
    concated_v_weight = torch.concat([weights[f"v_heads.{i}.weight"] for i in range(num_heads)], dim=0)

    query = in_features @ concated_q_weight.T   # (batch_size, seq_length, d_model) * (d_model, head * d_key) -> (batch_size, seq_length, head * d_key)
    key = in_features @ concated_k_weight.T
    value = in_features @ concated_v_weight.T

    seq_length = in_features.shape[-2]
    casual_mask = torch.tril(torch.ones(seq_length, seq_length)).bool()

    output = run_scaled_dot_product_attention(key, query, value, casual_mask, attn_pdrop)   #(batch_size, seq_length, head * d_key)
    output = output @ weights["output_proj.weight"].T

    return output
    

def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    attn_pdrop: float,
    residual_pdrop: float,
    weights: dict[str, torch.FloatTensor],
    in_features: torch.FloatTensor,
) -> torch.FloatTensor:
    """Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    Args:
        d_model: int
            The dimensionality of the Transformer block input.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff: int
            Dimensionality of the feed-forward inner layer (section 3.3).
        attn_pdrop: float
            Drop-out the attention probabilities (the softmax-normalized
            attention scores) with this rate.
        residual_pdrop: float
            Apply dropout to the output of each sub-layer, before it
            is added to the sub-layer input and normalized (section 5.4).
        weights: dict[str, torch.FloatTensor]
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, (d_model / num_heads) * num_heads).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features: torch.FloatTensor
            Tensor to run your implementation on.
            Shape is (batch_size, sequence_length, d_model).

    Returns:
        FloatTensor of shape (batch_size, sequence_length, d_model) with the output of
        running the Transformer block on the input features.
    """
    raise NotImplementedError


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    attn_pdrop: float,
    residual_pdrop: float,
    weights: dict[str, torch.FloatTensor],
    in_indices: torch.LongTensor,
) -> torch.FloatTensor:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    Args:
        vocab_size: int
            The number of unique items in the output vocabulary to be predicted.
        context_length: int,
            The maximum number of tokens to process at once.
        d_model: int
            The dimensionality of the model embeddings and sublayer outputs.
        num_layers: int
            The number of Transformer layers to use.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff: int
            Dimensionality of the feed-forward inner layer (section 3.3).
        attn_pdrop: float
            Drop-out the attention probabilities (the softmax-normalized
            attention scores) with this rate.
        residual_pdrop: float
            Apply dropout to the sum of the token and position embeddings
            as well as the output of each sub-layer, before it is added to the
            sub-layer input and normalized (section 5.4).
        weights: dict[str, torch.FloatTensor]
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `position_embeddings.weight`
                Positional embedding matrix. Shape is (context_length, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices: torch.LongTensor
            Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        FloatTensor of shape (batch size, sequence_length, vocab_size) with the predicted unnormalized
        next-word distribution for each token.
    """
    raise NotImplementedError


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: dict[str, torch.FloatTensor],
    in_features: torch.FloatTensor,
) -> torch.FloatTensor:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model: int
            The dimensionality of the RMSNorm input.
        eps: float, default is 1e-5
            A value added to the denominator for numerical stability.
        weights: dict[str, torch.FloatTensor]
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `weight`
                Weights of the RMSNorm affine transform.
                Shape is (d_model,).
        in_features: torch.FloatTensor
            Input features to run RMSNorm on. Tensor of (*, d_model), where *
            can be an arbitrary number of dimensions with arbitrary values.

    Returns:
        FloatTensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """

    rms = RMSNorm(d_model=d_model, eps=eps)
    rms.load_state_dict(weights)

    output = rms(in_features)

    return output


def run_gelu(in_features: torch.FloatTensor) -> torch.FloatTensor:
    """Given a tensor of inputs, return the output of applying GELU
    to each element.

    Args:
        in_features: torch.FloatTensor
            Input features to run GELU on. Shape is arbitrary.

    Returns:
        FloatTensor of with the same shape as `in_features` with the output of applying
        GELU to each element.
    """
    gelu = GELU()

    return gelu(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset: np.array
            1D numpy array of integer token IDs in the dataset.
        batch_size: int
            Desired batch size to sample.
        context_length: int
            Desired context length of each sampled example.
        device: str
            PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError


def run_softmax(in_features: torch.FloatTensor, dim: int) -> torch.FloatTensor:
    """Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features: torch.FloatTensor
            Input features to softmax. Shape is arbitrary.
        dim: int
            Dimension of the `in_features` to apply softmax to.

    Returns:
        FloatTensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    scaled_num = torch.unsqueeze(torch.max(in_features, axis=dim)[0], dim=-1)
    scaled_in_features = in_features - scaled_num

    exp_num = torch.exp(scaled_in_features)
    output = exp_num / torch.unsqueeze(torch.sum(exp_num, axis=dim), dim=dim)

    return output



def run_cross_entropy(inputs: torch.FloatTensor, targets: torch.LongTensor):
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs: torch.FloatTensor
            FloatTensor of shape (batch_size, num_classes). inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets: torch.LongTensor
            LongTensor of shape (batch_size, ) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Tensor of shape () with the average cross-entropy loss across examples.
    """
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters: collection of trainable parameters.
        max_l2_norm: a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.

    Returns:
        None
    """
    raise NotImplementedError


def get_adamw_cls() -> Type[torch.optim.Optimizer]:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it: int
            Iteration number to get learning rate for.
        max_learning_rate: float
            alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate: float
            alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters: int
            T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters: int
            T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model: torch.nn.Module
            Serialize the state of this model.
        optimizer: torch.optim.Optimizer,
            Serialize the state of this optimizer.
        iteration: int
            Serialize this value, which represents the number of training iterations
            we've completed.
        out: str | os.PathLike | BinaryIO | IO[bytes]
            Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src: str | os.PathLike | BinaryIO | IO[bytes]
            Path or file-like object to serialized checkpoint.
        model: torch.nn.Module
            Restore the state of this model.
        optimizer: torch.optim.Optimizer,
            Restore the state of this optimizer.
    Returns:
        int, the previously-serialized number of iterations.
    """
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: Optional[list[str]] = None,
):
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab: dict[int, bytes]
            The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges: list[tuple[bytes, bytes]]
            BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens: Optional[list[str]]
            A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    raise NotImplementedError


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
):
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path: str | os.PathLike
            Path to BPE tokenizer training data.
        vocab_size: int
            Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens: list[str]
            A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        Tuple of (vocab, merges):
            vocab: dict[int, bytes]
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges: list[tuple[bytes, bytes]]
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    # 1. pretoken and init vocab
    pre_tokenize_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    text = open(input_path, 'r', encoding='utf8').read()
    pretokens = Counter(re.findall(pre_tokenize_pattern, text))
    gen_tuple_of_bytes = lambda pretoken: tuple([bytes([b]) for b in pretoken.encode('utf8')])
    pretoken_freq = Counter()
    for pretoken, freq in pretokens.items():
        pretoken_freq[gen_tuple_of_bytes(pretoken)] = freq

    vocab = {i:bytes([i]) for i in range(256)}
    index = len(vocab)
    for token in special_tokens:
        vocab[index] = token.encode('utf8')
        index += 1

    # 2. pair frequence
    pair_freq = Counter()
    for pretoken, freq in pretoken_freq.items():
        for pair in zip(pretoken, pretoken[1:]):
            pair_freq[pair] = pair_freq.get(pair, 0) + freq
        

    # 3. trains
    merges = []
    while len(vocab) < vocab_size:
        most_freq_pair = max(pair_freq, key=lambda k:(pair_freq[k], k))

        merges.append(most_freq_pair)
        new_idx = max(vocab.keys()) + 1
        vocab[new_idx] = b"".join(most_freq_pair)

        new_pretoken_freq = {}
        for pretoken_tuple, freq in pretoken_freq.items():
            i = 0
            while i < len(pretoken_tuple):
                pair = pretoken_tuple[i:i+2]
                if pair == most_freq_pair:
                    pretoken_tuple, prefix, suffix = _update_byte_tuple(pretoken_tuple, i)

                    if prefix:
                        add_pair = (prefix[-1], vocab[new_idx])
                        pair_freq[add_pair] = pair_freq.get(add_pair, 0) + freq
                        del_pair = (prefix[-1], most_freq_pair[0])
                        pair_freq[del_pair] -= freq
                    
                    if suffix:
                        add_pair = (vocab[new_idx], suffix[0])
                        pair_freq[add_pair] = pair_freq.get(add_pair, 0) + freq
                        del_pair = (most_freq_pair[1], suffix[0])
                        pair_freq[del_pair] -= freq
                    
                    pair_freq[most_freq_pair] -= freq
                
                i += 1

            new_pretoken_freq[pretoken_tuple] = freq
        pretoken_freq = new_pretoken_freq
            
    return vocab, merges

def _update_byte_tuple(byte_tuple, merge_loc):
    assert len(byte_tuple) > 1, "byte tuple length <= 1, can't merge!"
    prefix = byte_tuple[:merge_loc]
    tomerge = byte_tuple[merge_loc:merge_loc+2]
    suffix = byte_tuple[merge_loc+2:]

    new_byte_tuple = prefix + (b"".join(tomerge),) + suffix

    return new_byte_tuple, prefix, suffix
