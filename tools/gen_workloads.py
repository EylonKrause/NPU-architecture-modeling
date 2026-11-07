#!/usr/bin/env python3
"""Generate the canonical workload layer tables in workloads/ (one JSON file per workload).

    python tools/gen_workloads.py            rewrite workloads/*.json
    python tools/gen_workloads.py --check    exit 1 if the committed files differ

Conventions (also written into every file):
- batch 1, NHWC, INT8 weights (1 byte each), INT32 bias reported separately, BN folded.
- conv MACs = out_h * out_w * out_c * kernel_h * kernel_w * (in_c / groups).
- "lowered" is the npu-sim call that runs the layer: an M x N x K GEMM per group, G groups
  (--groups G), repeated "calls" times. Convolutions lower by im2col: M = out_h * out_w,
  N = out_c / G, K = kernel_h * kernel_w * in_c / G; depthwise is G = in_c, N = channel
  multiplier, K = kernel_h * kernel_w. Attention matmuls lower per KV head (G = kv_heads)
  with the query heads that share a KV head stacked along M.
- Parameter totals are checked against the published counts cited in "source".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "workloads"

CONVENTIONS = {
    "batch": 1,
    "layout": "NHWC",
    "weight_bytes": "INT8, 1 byte per weight; INT32 bias reported separately; BN folded",
    "macs": "one multiply-accumulate; FLOPs = 2 x MACs",
    "repeat": "number of identical instances of the layer in the network",
    "lowered": "npu-sim call per instance: M x N x K GEMM per group, G groups (--groups), "
               "'calls' identical calls; conv via im2col, depthwise as G = channels, "
               "attention per KV head with sharing query heads stacked along M",
}


def conv_out(n, k, s, p_lo, p_hi):
    return (n + p_lo + p_hi - k) // s + 1


def conv(name, h, w, cin, cout, k, s, pad, pad_tblr=None, repeat=1, groups=1, bias=True, note=None):
    t, b, l, r = pad_tblr if pad_tblr else (pad, pad, pad, pad)
    oh, ow = conv_out(h, k, s, t, b), conv_out(w, k, s, l, r)
    dw = groups == cin and groups > 1
    cin_g = cin // groups
    macs = oh * ow * cout * k * k * cin_g
    layer = {"name": name, "op": "depthwise_conv2d" if dw else "conv2d",
             "in_h": h, "in_w": w, "in_c": cin, "out_c": cout,
             "kernel_h": k, "kernel_w": k, "stride": s, "pad": pad,
             "groups": groups, "repeat": repeat, "out_h": oh, "out_w": ow,
             "macs": macs, "weight_bytes_int8": k * k * cin_g * cout,
             "bias_bytes_int32": 4 * cout if bias else 0,
             "lowered": {"M": oh * ow, "N": cout // groups, "K": k * k * cin_g, "G": groups,
                         "calls": repeat}}
    if pad_tblr:
        layer["pad_tblr"] = list(pad_tblr)
    if note:
        layer["note"] = note
    return layer


def mm(name, m, n, k, repeat=1, weights=True, bias=False, note=None, op="dense", lowered=None):
    layer = {"name": name, "op": op if weights else "matmul", "M": m, "N": n, "K": k,
             "repeat": repeat, "macs": m * n * k,
             "weight_bytes_int8": k * n if weights else 0,
             "bias_bytes_int32": 4 * n if (weights and bias) else 0,
             "b_operand": "weights" if weights else "activations",
             "lowered": lowered or {"M": m, "N": n, "K": k, "G": 1, "calls": repeat}}
    if note:
        layer["note"] = note
    return layer


def attention(name, m, n, k, heads, kv_heads, layers, note=None):
    """Per-query-head matmul run as one grouped call per layer: G = kv_heads, each group
    stacks the heads that share its K/V along M."""
    share = heads // kv_heads
    lowered = {"M": share * m, "N": n, "K": k, "G": kv_heads, "calls": layers}
    return mm(name, m, n, k, repeat=layers * heads, weights=False, note=note, lowered=lowered)


def mobilenet_v2():
    L, V = [], []
    # Keras: Conv1 'same' s2 on 224 -> TF SAME pads (0,1,0,1); torchvision pads 1 all round.
    L.append(conv("Conv1", 224, 224, 3, 32, 3, 2, 1, pad_tblr=(0, 1, 0, 1), bias=True,
                  note="TF SAME padding (0,1,0,1); torchvision uses pad=1, identical output size"))
    V.append({"name": "Conv1_relu6", "op": "relu6", "elements": 112 * 112 * 32, "repeat": 1})
    # (t, c, n, s) from Table 2 of arXiv:1801.04381
    cfg = [(1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1),
           (6, 160, 3, 2), (6, 320, 1, 1)]
    h, cin, bid = 112, 32, 0
    for t, c, n, s in cfg:
        for i in range(n):
            stride = s if i == 0 else 1
            prefix = "block_%d" % bid if bid else "expanded_conv"
            hid = cin * t
            if t != 1:
                L.append(conv(prefix + "_expand", h, h, cin, hid, 1, 1, 0))
                V.append({"name": prefix + "_expand_relu6", "op": "relu6",
                          "elements": h * h * hid, "repeat": 1})
            if stride == 2:
                # Keras correct_pad for even input: ZeroPadding2D(((0,1),(0,1))) then 'valid'
                L.append(conv(prefix + "_depthwise", h, h, hid, hid, 3, 2, 1, pad_tblr=(0, 1, 0, 1),
                              groups=hid, note="Keras correct_pad (0,1,0,1)+valid; torchvision "
                                               "pad=1; identical output size"))
            else:
                L.append(conv(prefix + "_depthwise", h, h, hid, hid, 3, 1, 1, groups=hid))
            ho = L[-1]["out_h"]
            V.append({"name": prefix + "_depthwise_relu6", "op": "relu6",
                      "elements": ho * ho * hid, "repeat": 1})
            L.append(conv(prefix + "_project", ho, ho, hid, c, 1, 1, 0,
                          note="linear bottleneck (no activation)"))
            if stride == 1 and cin == c:
                V.append({"name": prefix + "_add", "op": "residual_add",
                          "elements": ho * ho * c, "repeat": 1})
            h, cin, bid = ho, c, bid + 1
    L.append(conv("Conv_1", 7, 7, 320, 1280, 1, 1, 0))
    V.append({"name": "Conv_1_relu6", "op": "relu6", "elements": 7 * 7 * 1280, "repeat": 1})
    V.append({"name": "global_average_pooling", "op": "avgpool_7x7", "elements": 7 * 7 * 1280,
              "repeat": 1})
    L.append(mm("Logits", 1, 1000, 1280, bias=True,
                note="Keras Dense(1000); paper Table 2 writes it as conv2d 1x1 1280->k"))
    V.append({"name": "softmax", "op": "softmax", "elements": 1000, "repeat": 1})
    return L, V


def resnet50_v15():
    L, V = [], []
    L.append(conv("conv1", 224, 224, 3, 64, 7, 2, 3))
    V.append({"name": "conv1_relu", "op": "relu", "elements": 112 * 112 * 64, "repeat": 1})
    V.append({"name": "maxpool", "op": "maxpool_3x3_s2_p1", "elements": 56 * 56 * 64, "repeat": 1})
    h, cin = 56, 64
    for stage, (width, blocks, stride) in enumerate([(64, 3, 1), (128, 4, 2), (256, 6, 2),
                                                     (512, 3, 2)], start=1):
        out = width * 4
        p = "layer%d.0" % stage
        L.append(conv(p + ".conv1", h, h, cin, width, 1, 1, 0,
                      note="v1.5: stride 1 here, at the INPUT resolution" if stride == 2 else None))
        L.append(conv(p + ".conv2", h, h, width, width, 3, stride, 1,
                      note="v1.5: stride lives in the 3x3" if stride == 2 else None))
        ho = L[-1]["out_h"]
        L.append(conv(p + ".conv3", ho, ho, width, out, 1, 1, 0))
        L.append(conv(p + ".downsample", h, h, cin, out, 1, stride, 0))
        V.append({"name": p + ".relu", "op": "relu",
                  "elements": h * h * width + 2 * ho * ho * width + ho * ho * out, "repeat": 1})
        V.append({"name": p + ".add", "op": "residual_add", "elements": ho * ho * out, "repeat": 1})
        h, cin = ho, out
        q = "layer%d.1-%d" % (stage, blocks - 1)
        r = blocks - 1
        L.append(conv(q + ".conv1", h, h, out, width, 1, 1, 0, repeat=r))
        L.append(conv(q + ".conv2", h, h, width, width, 3, 1, 1, repeat=r))
        L.append(conv(q + ".conv3", h, h, width, out, 1, 1, 0, repeat=r))
        V.append({"name": q + ".relu", "op": "relu",
                  "elements": 2 * h * h * width + h * h * out, "repeat": r})
        V.append({"name": q + ".add", "op": "residual_add", "elements": h * h * out, "repeat": r})
    V.append({"name": "avgpool", "op": "avgpool_7x7", "elements": 7 * 7 * 2048, "repeat": 1})
    L.append(mm("fc", 1, 1000, 2048, bias=True))
    V.append({"name": "softmax", "op": "softmax", "elements": 1000, "repeat": 1})
    return L, V


LLAMA = dict(hidden=2048, intermediate=8192, heads=32, kv_heads=8, head_dim=64, layers=16,
             vocab=128256, tied=True)


def llama(m, kv_len, phase):
    c = LLAMA
    d, f, H, KV, hd, nl, voc = (c["hidden"], c["intermediate"], c["heads"], c["kv_heads"],
                                c["head_dim"], c["layers"], c["vocab"])
    L, V = [], []
    L.append(mm("q_proj", m, H * hd, d, repeat=nl))
    L.append(mm("k_proj", m, KV * hd, d, repeat=nl, note="GQA: 8 KV heads x 64 = 512"))
    L.append(mm("v_proj", m, KV * hd, d, repeat=nl, note="GQA: 8 KV heads x 64 = 512"))
    causal = (" Causal mask: only %d of %d score entries are live (%.1f%%)."
              % (m * (m + 1) // 2, m * kv_len, 100.0 * m * (m + 1) / 2 / (m * kv_len))
              if phase == "prefill" else "")
    L.append(attention("attn_scores_QKt", m, kv_len, hd, H, KV, nl,
                       note="per query head; 4 query heads share each KV head." + causal))
    L.append(attention("attn_context_PV", m, hd, kv_len, H, KV, nl,
                       note="per query head; 4 query heads share each KV head"))
    L.append(mm("o_proj", m, d, H * hd, repeat=nl))
    L.append(mm("gate_proj", m, f, d, repeat=nl))
    L.append(mm("up_proj", m, f, d, repeat=nl))
    L.append(mm("down_proj", m, d, f, repeat=nl))
    L.append(mm("lm_head", 1, voc, d, repeat=1,
                note="tied with embed_tokens (weights counted once in the total). Prefill "
                     "computes logits for the last position only, as serving runtimes do; HF "
                     "default full-sequence logits would be M=%d" % m))
    V += [{"name": "input_layernorm_rmsnorm", "op": "rmsnorm", "elements": m * d, "repeat": nl},
          {"name": "rope_qk", "op": "rope", "elements": m * (H + KV) * hd, "repeat": nl},
          {"name": "softmax", "op": "softmax", "elements": H * m * kv_len, "repeat": nl},
          {"name": "residual_add_attn", "op": "residual_add", "elements": m * d, "repeat": nl},
          {"name": "post_attention_layernorm_rmsnorm", "op": "rmsnorm", "elements": m * d,
           "repeat": nl},
          {"name": "silu_gate_x_up", "op": "silu_mul", "elements": m * f, "repeat": nl},
          {"name": "residual_add_mlp", "op": "residual_add", "elements": m * d, "repeat": nl},
          {"name": "final_norm_rmsnorm", "op": "rmsnorm", "elements": m * d, "repeat": 1},
          {"name": "embedding_gather", "op": "gather", "elements": m * d, "repeat": 1}]
    return L, V


BERT = dict(hidden=768, intermediate=3072, heads=12, head_dim=64, layers=12, vocab=30522,
            max_pos=512, type_vocab=2)


def bert_layer(s=128):
    c = BERT
    d, f, H, hd = c["hidden"], c["intermediate"], c["heads"], c["head_dim"]
    L, V = [], []
    L.append(mm("attention.self.query", s, d, d, bias=True))
    L.append(mm("attention.self.key", s, d, d, bias=True))
    L.append(mm("attention.self.value", s, d, d, bias=True,
                note="Q/K/V often fused as one N=2304 GEMM (226,492,416 MACs)"))
    L.append(attention("attn_scores_QKt", s, s, hd, H, H, 1,
                       note="bidirectional: full SxS, no causal mask"))
    L.append(attention("attn_context_PV", s, hd, s, H, H, 1))
    L.append(mm("attention.output.dense", s, d, d, bias=True))
    L.append(mm("intermediate.dense", s, f, d, bias=True))
    L.append(mm("output.dense", s, d, f, bias=True))
    V += [{"name": "softmax", "op": "softmax", "elements": H * s * s, "repeat": 1},
          {"name": "attention.output.LayerNorm(+residual)", "op": "layernorm", "elements": s * d,
           "repeat": 1},
          {"name": "intermediate.gelu", "op": "gelu", "elements": s * f, "repeat": 1},
          {"name": "output.LayerNorm(+residual)", "op": "layernorm", "elements": s * d,
           "repeat": 1}]
    return L, V


def totals(layers):
    lowered = sum(x["lowered"]["M"] * x["lowered"]["N"] * x["lowered"]["K"] * x["lowered"]["G"]
                  * x["lowered"]["calls"] for x in layers)
    macs = sum(x["macs"] * x["repeat"] for x in layers)
    assert lowered == macs, "lowered GEMMs must cover every MAC exactly"
    return {"macs": macs,
            "weight_bytes_int8": sum(x["weight_bytes_int8"] * x["repeat"] for x in layers),
            "bias_bytes_int32": sum(x["bias_bytes_int32"] * x["repeat"] for x in layers),
            "simulator_calls": sum(x["lowered"]["calls"] for x in layers)}


def build():
    W = []
    L, V = mobilenet_v2()
    t = totals(L)
    bn_params = sum(x["out_c"] for x in L if x["op"] in ("conv2d", "depthwise_conv2d")) * 2
    tv_params = t["weight_bytes_int8"] + bn_params + 1000
    t["param_check"] = {"conv_fc_weights": t["weight_bytes_int8"], "bn_gamma_beta": bn_params,
                        "fc_bias": 1000, "computed_torchvision_style_params": tv_params,
                        "published_torchvision_params": 3504872, "match": tv_params == 3504872}
    t["reference"] = {"published_macs": 300_000_000,
                      "published_source": "Sandler et al. 2018, Table 4: 300M MAdds, 3.4M params; "
                                          "torchvision MobileNet_V2_Weights.IMAGENET1K_V1: 0.30 "
                                          "GFLOPS (MACs), 3,504,872 params",
                      "computed_vs_published_macs": round(t["macs"] / 300_000_000, 4)}
    W.append({"name": "mobilenet_v2_1.0_224",
              "source": ["https://arxiv.org/abs/1801.04381",
                         "https://github.com/keras-team/keras/blob/master/keras/src/applications/mobilenet_v2.py",
                         "https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v2.html"],
              "notes": "alpha=1.0, 224x224x3, batch 1. Table 2 (t,c,n,s): (1,16,1,1) (6,24,2,2) "
                       "(6,32,3,2) (6,64,4,2) (6,96,3,1) (6,160,3,2) (6,320,1,1). The first block "
                       "(t=1) has no expansion conv. Stride applies to the first block of each "
                       "sequence, in the 3x3 depthwise. Expansion t multiplies the block INPUT "
                       "channels. BN folded into bias; ReLU6 after expand and depthwise, linear "
                       "after project. Residual add when stride 1 and in_c == out_c (10 adds).",
              "layers": L, "vector_ops": V, "totals": t})
    L, V = resnet50_v15()
    t = totals(L)
    bn_params = sum(x["out_c"] * x["repeat"] for x in L if x["op"] == "conv2d") * 2
    tv_params = t["weight_bytes_int8"] + bn_params + 1000
    # v1 puts stride 2 in the 1x1 conv1 of layer2-4 block 1: that conv runs at output resolution.
    v1_delta = sum(x["macs"] * 3 // 4 for x in L
                   if x["name"] in ("layer2.0.conv1", "layer3.0.conv1", "layer4.0.conv1"))
    t["param_check"] = {"conv_fc_weights": t["weight_bytes_int8"], "bn_gamma_beta": bn_params,
                        "fc_bias": 1000, "computed_torchvision_style_params": tv_params,
                        "published_torchvision_params": 25557032, "match": tv_params == 25557032}
    t["reference"] = {"published_macs": 4_090_000_000,
                      "published_source": "torchvision ResNet50_Weights: 4.09 GFLOPS (fvcore counts "
                                          "MACs), 25,557,032 params; He et al. 2015 Table 1: 3.8e9 "
                                          "FLOPs (multiply-adds) for v1",
                      "computed_vs_published_macs": round(t["macs"] / 4_090_000_000, 4),
                      "v1_macs_computed": t["macs"] - v1_delta,
                      "v1_vs_paper_3.8e9": round((t["macs"] - v1_delta) / 3.8e9, 4),
                      "v1_5_extra_macs_over_v1": v1_delta}
    W.append({"name": "resnet50_v1.5_224",
              "source": ["https://arxiv.org/abs/1512.03385",
                         "https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html",
                         "https://catalog.ngc.nvidia.com/orgs/nvidia/resources/resnet_50_v1_5_for_pytorch"],
              "notes": "224x224x3, batch 1. Stages (bottleneck width, blocks, stride): (64,3,1) "
                       "(128,4,2) (256,6,2) (512,3,2); expansion 4. v1.5 = stride in the 3x3 conv2, "
                       "not the first 1x1 (torchvision/NVIDIA/MLPerf). Projection shortcut (1x1, "
                       "stride s) on the first block of every stage. BN folded into bias. 7x7/s2 "
                       "stem + 3x3/s2 maxpool. Layers with repeat > 1 are the identical blocks 2..n "
                       "of a stage.",
              "layers": L, "vector_ops": V, "totals": t})
    c = LLAMA
    per_layer_w = (c["hidden"] * c["heads"] * c["head_dim"] * 2
                   + c["hidden"] * c["kv_heads"] * c["head_dim"] * 2
                   + 3 * c["hidden"] * c["intermediate"])
    total_params = c["layers"] * (per_layer_w + 2 * c["hidden"]) + c["vocab"] * c["hidden"] + c["hidden"]
    src = ["https://huggingface.co/unsloth/Llama-3.2-1B/raw/main/config.json (ungated mirror of "
           "meta-llama/Llama-3.2-1B config.json)",
           "https://huggingface.co/meta-llama/Llama-3.2-1B"]
    cfg_note = ("config.json: hidden_size 2048, intermediate_size 8192, num_attention_heads 32, "
                "num_key_value_heads 8, head_dim 64, num_hidden_layers 16, vocab_size 128256, "
                "tie_word_embeddings true, no attention/MLP bias, SiLU-gated MLP, RMSNorm, RoPE "
                "(llama3 scaling, theta 5e5). ")
    for phase, m, kv in (("prefill", 128, 128), ("decode", 1, 129)):
        L, V = llama(m, kv, phase)
        t = totals(L)
        t["param_check"] = {"computed_total_params": total_params,
                            "published_params": "1.23B (model card); HF 1,235,814,400",
                            "match": total_params == 1_235_814_400,
                            "note": "weight_bytes_int8 counts lm_head (= tied embed_tokens) once; "
                                    "RMSNorm gains excluded"}
        t["kv_cache_bytes_int8"] = 2 * c["layers"] * kv * c["kv_heads"] * c["head_dim"]
        approx = c["layers"] * per_layer_w * m + c["vocab"] * c["hidden"]
        t["reference"] = {"rule_of_thumb": "MACs ~= non-embedding params x tokens + attention + lm_head",
                          "non_embedding_params_x_tokens_plus_lm_head": approx,
                          "computed_vs_rule": round(t["macs"] / approx, 4)}
        if phase == "decode":
            t["arithmetic_intensity_macs_per_weight_byte"] = round(t["macs"] / t["weight_bytes_int8"], 4)
        W.append({"name": "llama3.2_1b_%s_%s" % (phase, "s128" if phase == "prefill" else "kv129"),
                  "source": src,
                  "notes": cfg_note + ("Prefill of a 128-token prompt, batch 1: every projection "
                                       "has M=128. " if phase == "prefill" else
                                       "Decode of the first generated token after a 128-token "
                                       "prompt (KV length 129), batch 1: every projection is a "
                                       "GEMV (M=1), so the step is weight-bandwidth bound. ")
                  + "Attention matmuls (b_operand=activations) read the KV cache, not weights. "
                    "repeat = 16 layers (x32 heads for attention matmuls). The INT8 KV cache for "
                    "this context is in totals.",
                  "layers": L, "vector_ops": V, "totals": t})
    L, V = bert_layer(128)
    t = totals(L)
    b = BERT
    per_layer_params = (4 * (b["hidden"] ** 2 + b["hidden"]) + 2 * b["hidden"] * b["intermediate"]
                        + b["intermediate"] + b["hidden"] + 4 * b["hidden"])
    embed = (b["vocab"] + b["max_pos"] + b["type_vocab"]) * b["hidden"] + 2 * b["hidden"]
    pooler = b["hidden"] ** 2 + b["hidden"]
    full = b["layers"] * per_layer_params + embed + pooler
    t["param_check"] = {"per_layer_params_hf": per_layer_params, "computed_bert_base_params": full,
                        "published_params": "110M (Devlin et al.); HF bert-base-uncased 109,482,240",
                        "match": full == 109_482_240}
    t["encoder_x12"] = {"macs": t["macs"] * 12, "weight_bytes_int8": t["weight_bytes_int8"] * 12}
    t["reference"] = {"published_flops": 22_500_000_000,
                      "published_source": "Sun et al. 2020 (MobileBERT) Table 4: BERT_BASE 109M "
                                          "params, 22.5B FLOPS at seq 128",
                      "computed_encoder_x12_flops_2_per_mac": t["macs"] * 24,
                      "computed_vs_published": round(t["macs"] * 24 / 22.5e9, 4)}
    W.append({"name": "bert_base_encoder_layer_s128",
              "source": ["https://huggingface.co/google-bert/bert-base-uncased/raw/main/config.json",
                         "https://arxiv.org/abs/1810.04805", "https://arxiv.org/abs/2004.02984"],
              "notes": "One BERT-base encoder layer at sequence 128, batch 1: hidden 768, 12 heads "
                       "x 64, FFN 3072, GELU, post-LN, biases on every projection. The full encoder "
                       "is this layer x12 (totals.encoder_x12). Attention is bidirectional.",
              "layers": L, "vector_ops": V, "totals": t})
    return [{"schema_version": 1, "conventions": CONVENTIONS, **w} for w in W]


def render(workload) -> str:
    return json.dumps(workload, indent=1) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="gen_workloads", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="compare instead of writing")
    parser.add_argument("--out", type=Path, default=OUT, help="output directory")
    args = parser.parse_args(argv)
    stale = []
    for workload in build():
        path = args.out / (workload["name"] + ".json")
        text = render(workload)
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                stale.append(path.name)
        else:
            args.out.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="\n")
    if stale:
        print("gen_workloads: out of date: " + ", ".join(stale), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
