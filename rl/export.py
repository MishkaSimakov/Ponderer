"""SB3 policy to a single npz the brick can read.

Only the actor path is exported. The critic and log_std exist to train and stay on
the host; on the brick the policy is deterministic, so the action is the clipped mean.

Layer names here are the contract with shared/policies/net.py.
"""

import os

import numpy as np
from torch import nn

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "policy")


def _put(out, name, tensor):
    out[name] = tensor.detach().cpu().numpy().astype(np.float32)


def _linear(out, name, layer):
    _put(out, name + ".w", layer.weight)
    _put(out, name + ".b", layer.bias)


def _head(out, policy):
    """mlp_extractor.policy_net, then action_net. Tanh between the linears."""
    linears = []
    for module in policy.mlp_extractor.policy_net:
        if isinstance(module, nn.Linear):
            linears.append(module)
        elif not isinstance(module, nn.Tanh):
            raise ValueError("head activation is %s, net.py only does tanh" % type(module))

    for i, layer in enumerate(linears):
        _linear(out, "pi.%d" % i, layer)
    out["pi_layers"] = np.array(len(linears))
    _linear(out, "action", policy.action_net)


def _mlp(out, policy):
    _head(out, policy)


def _lstm(out, policy):
    lstm = policy.lstm_actor
    if lstm.num_layers != 1:
        raise ValueError("net.py runs a single lstm layer, got %d" % lstm.num_layers)

    # nn.LSTM keeps all four gates in one matrix, in the order i, f, g, o. Both biases
    # are added, exactly as torch does.
    _put(out, "lstm.ih.w", lstm.weight_ih_l0)
    _put(out, "lstm.ih.b", lstm.bias_ih_l0)
    _put(out, "lstm.hh.w", lstm.weight_hh_l0)
    _put(out, "lstm.hh.b", lstm.bias_hh_l0)
    out["hidden"] = np.array(lstm.hidden_size)
    _head(out, policy)


def _transformer(out, policy):
    extractor = policy.features_extractor
    _linear(out, "enc.project", extractor.project)
    _put(out, "enc.pos", extractor.position)
    out["window"] = np.array(extractor.window)
    out["frame"] = np.array(extractor.frame)
    out["layers"] = np.array(len(extractor.encoder.layers))

    for i, layer in enumerate(extractor.encoder.layers):
        if layer.norm_first:
            raise ValueError("net.py implements post norm blocks")

        attn = layer.self_attn
        out["heads"] = np.array(attn.num_heads)
        _put(out, "enc.%d.qkv.w" % i, attn.in_proj_weight)
        _put(out, "enc.%d.qkv.b" % i, attn.in_proj_bias)
        _linear(out, "enc.%d.proj" % i, attn.out_proj)
        _linear(out, "enc.%d.ff1" % i, layer.linear1)
        _linear(out, "enc.%d.ff2" % i, layer.linear2)
        _put(out, "enc.%d.norm1.w" % i, layer.norm1.weight)
        _put(out, "enc.%d.norm1.b" % i, layer.norm1.bias)
        _put(out, "enc.%d.norm2.w" % i, layer.norm2.weight)
        _put(out, "enc.%d.norm2.b" % i, layer.norm2.bias)

    _head(out, policy)


WRITERS = {"mlp": _mlp, "lstm": _lstm, "transformer": _transformer}


def export(model, arch, name):
    """Write policy/<name>.npz. sync.sh copies that directory to the brick."""
    out = {"arch": np.array(arch)}
    WRITERS[arch](out, model.policy)

    os.makedirs(ROOT, exist_ok=True)
    path = os.path.join(ROOT, name + ".npz")
    np.savez(path, **out)
    return path
