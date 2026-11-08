"""Run an INT8 graph on the NPU model: Dense/MatMul/Conv2D/DepthwiseConv2D on the accelerator
(convolutions via im2col; depthwise as one grouped call when npu-sim has --groups),
Add/ReLU/pooling/Flatten on the host. Every accelerator result is checked against a scalar
Python reference before the next node runs.
"""

from __future__ import annotations

import math
import tempfile
import time
from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from . import cli, jsonio, npufile, sim, table
from . import quant as rq

MAX_ELEMENTS = 1_000_000
MAX_NODES = 256
INT32_MIN, INT32_MAX = rq.INT32_MIN, rq.INT32_MAX
# Flags the runtime sets per call; a graph's "architecture" object may set any other
# npu-sim flag (tile_m, macs_per_cycle, dataflow, memory_model, ...).
RUNTIME_FLAGS = {
    "m",
    "n",
    "k",
    "groups",
    "input",
    "dump_output",
    "input_zero",
    "weight_zero",
    "output_zero",
    "multiplier",
    "shift",
    "rounding",
    "rescale",
    "relu",
    "quantization_file",
    "seed",
    "trace",
    "vcd",
    "progress_jsonl",
    "slice_cycles",
    "live_stdio",
    "config",
    "timing_only",
    "help",
    "version",
    "describe_config",
}


def integer(value, name, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(name + " must be an integer")
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(name + " is outside the supported range")
    return value


def scale_value(value, name="scale"):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + " must be finite and positive")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(name + " must be finite and positive") from error
    if not math.isfinite(result) or result <= 0:
        raise ValueError(name + " must be finite and positive")
    return result


def shape_value(value):
    if not isinstance(value, list) or not value or len(value) > 6:
        raise ValueError("shape must have one through six dimensions")
    shape = tuple(integer(x, "dimension", 1, 4096) for x in value)
    if math.prod(shape) > MAX_ELEMENTS:
        raise ValueError("tensor exceeds graph element limit")
    return shape


@dataclass
class Tensor:
    shape: tuple
    data: list
    dtype: str = "int8"
    scale: float = 1.0
    zero_point: int = 0

    @property
    def bytes(self):
        return len(self.data) * (4 if self.dtype == "int32" else 1)

    def descriptor(self, include_data=False):
        result = {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "scale": self.scale,
            "zero_point": self.zero_point,
        }
        if include_data:
            result["data"] = self.data
        return result


def parse_tensor(spec):
    if not isinstance(spec, dict):
        raise ValueError("tensor must be an object")
    unknown = set(spec) - {"dtype", "shape", "data", "scale", "zero_point"}
    if unknown:
        raise ValueError("unknown tensor fields: " + ", ".join(sorted(unknown)))
    if not {"dtype", "shape", "data", "scale", "zero_point"} <= set(spec):
        raise ValueError("tensor requires dtype, shape, data, scale and zero_point")
    dtype = spec["dtype"]
    if dtype not in ("int8", "int32"):
        raise ValueError("only int8 and int32 tensors are supported")
    shape = shape_value(spec["shape"])
    values = spec["data"]
    if not isinstance(values, list) or len(values) != math.prod(shape):
        raise ValueError("flat data length does not match tensor shape")
    lo, hi = (-128, 127) if dtype == "int8" else (INT32_MIN, INT32_MAX)
    data = [integer(x, "tensor value", lo, hi) for x in values]
    zero = integer(spec["zero_point"], "zero_point", lo, hi)
    if dtype == "int32" and zero != 0:
        raise ValueError("INT32 bias zero_point must be zero")
    return Tensor(shape, data, dtype, scale_value(spec["scale"]), zero)


round_div = rq.round_div
saturate = rq.saturate


def channel_quant(q, channel):
    return {**q, **q["channels"][channel]} if "channels" in q else q


def requantize(value, q):
    integer(value, "accumulator", INT32_MIN, INT32_MAX)
    return rq.requantize(value, q)


def quantization(node, activation, weights, bias=None, channels=1):
    integer(channels, "output channels", 1, 4096)
    raw = node.get("quant")
    if not isinstance(raw, dict):
        raise ValueError("accelerated nodes require a quant object")
    if set(raw) - {
        "multiplier",
        "shift",
        "output_scale",
        "output_zero",
        "relu",
        "weight_zero_points",
        "multipliers",
        "shifts",
        "weight_scales",
        "bias_scales",
        "rounding",
        "rescale",
    }:
        raise ValueError("unsupported quantization field")
    if not {"multiplier", "shift", "output_scale", "output_zero"} <= set(raw):
        raise ValueError("quant requires multiplier, shift, output_scale and output_zero")
    mode = raw.get("rescale", "native")
    if mode not in rq.RESCALE_MODES:
        raise ValueError("rescale must be native, tosa-single or tosa-double")
    low_shift, high_shift = (0, 31) if mode == "native" else (2, 62)
    q = {
        "multiplier": integer(raw["multiplier"], "multiplier", 0, INT32_MAX),
        "shift": integer(raw["shift"], "shift", low_shift, high_shift),
        "output_zero": integer(raw["output_zero"], "output_zero", -128, 127),
        "output_scale": scale_value(raw["output_scale"], "output_scale"),
        "relu": raw.get("relu", False),
        "rounding": raw.get("rounding", "away"),
        "rescale": mode,
    }
    if q["rounding"] not in ("away", "even"):
        raise ValueError("rounding must be away or even")
    if not isinstance(q["relu"], bool):
        raise ValueError("relu must be boolean")
    fields = {
        "weight_zero_points": ("weight_zero", weights.zero_point, -128, 127),
        "multipliers": ("multiplier", q["multiplier"], 0, INT32_MAX),
        "shifts": ("shift", q["shift"], low_shift, high_shift),
    }
    lanes = [
        {"weight_zero": weights.zero_point, "multiplier": q["multiplier"], "shift": q["shift"]}
        for _ in range(channels)
    ]
    for key, (destination, default, lo, hi) in fields.items():
        values = raw.get(key, [default] * channels)
        if not isinstance(values, list) or len(values) != channels:
            raise ValueError(key + " must contain one entry per output channel")
        for index, value in enumerate(values):
            lanes[index][destination] = integer(value, key, lo, hi)
    weight_scales = raw.get("weight_scales", [weights.scale] * channels)
    bias_scales = raw.get("bias_scales", [bias.scale if bias is not None else 1] * channels)
    for key, values in (("weight_scales", weight_scales), ("bias_scales", bias_scales)):
        if not isinstance(values, list) or len(values) != channels:
            raise ValueError(key + " must contain one entry per output channel")
        for value in values:
            scale_value(value, key)
    if bias is not None and (bias.dtype != "int32" or bias.zero_point != 0):
        raise ValueError("bias must be INT32 with zero_point zero")
    for index, lane in enumerate(lanes):
        real_ratio = activation.scale * weight_scales[index] / q["output_scale"]
        represented = lane["multiplier"] / (1 << lane["shift"])
        if (
            not math.isfinite(real_ratio)
            or abs(real_ratio - represented) > 0.5 / (1 << lane["shift"]) + abs(real_ratio) * 1e-12
        ):
            raise ValueError("integer requantization does not represent the declared tensor scales")
        if bias is not None and not math.isclose(
            bias_scales[index], activation.scale * weight_scales[index], rel_tol=1e-12, abs_tol=0
        ):
            raise ValueError("bias scale must equal activation scale times weight scale")
    if any(key in raw for key in (*fields, "weight_scales", "bias_scales")):
        q["channels"] = lanes
    return q


def require_int8(*tensors):
    if any(x.dtype != "int8" for x in tensors):
        raise ValueError("operator inputs must be INT8")


def pair(value, name, default):
    if value is None:
        value = default
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(name + " must contain two positive integers")
    return tuple(integer(x, name, 1, 4096) for x in value)


def spatial(node, height, width, kh, kw):
    sh, sw = pair(node.get("strides"), "strides", [1, 1])
    dh, dw = pair(node.get("dilations"), "dilations", [1, 1])
    eh, ew = (kh - 1) * dh + 1, (kw - 1) * dw + 1
    pad = node.get("padding", "VALID")
    if pad == "SAME":
        oh, ow = (height + sh - 1) // sh, (width + sw - 1) // sw
        ph, pw = max(0, (oh - 1) * sh + eh - height), max(0, (ow - 1) * sw + ew - width)
        top, left = ph // 2, pw // 2
    elif pad == "VALID":
        top = left = 0
        oh, ow = (height - eh) // sh + 1, (width - ew) // sw + 1
    elif isinstance(pad, list) and len(pad) == 4:
        top, bottom, left, right = [integer(x, "padding", 0, 4096) for x in pad]
        oh, ow = (height + top + bottom - eh) // sh + 1, (width + left + right - ew) // sw + 1
    else:
        raise ValueError("padding must be SAME, VALID or [top,bottom,left,right]")
    if oh < 1 or ow < 1:
        raise ValueError("kernel and padding produce an empty output")
    return oh, ow, sh, sw, dh, dw, top, left


def nhwc_index(shape, b, y, x, c):
    return ((b * shape[1] + y) * shape[2] + x) * shape[3] + c


def dense_reference(a, w, bias, q):
    m, k = a.shape
    n = w.shape[1]
    accumulators, outputs = [], []
    for row in range(m):
        for col in range(n):
            total = bias[col]
            lane = channel_quant(q, col)
            weight_zero = lane.get("weight_zero", w.zero_point)
            for reduction in range(k):
                total += (a.data[row * k + reduction] - a.zero_point) * (
                    w.data[reduction * n + col] - weight_zero
                )
                if not -(1 << 63) <= total < (1 << 63):
                    raise ValueError("reference accumulation exceeds INT64")
            integer(total, "accumulator", INT32_MIN, INT32_MAX)
            accumulators.append(total)
            outputs.append(requantize(total, lane))
    return accumulators, outputs


def convolution_reference(a, w, bias, q, geometry, depthwise):
    """Direct NHWC scalar loops, independent of the im2col lowering the accelerator runs."""
    oh, ow, sh, sw, dh, dw, top, left = geometry
    batches, height, width, channels = a.shape
    kh, kw, _, last = w.shape
    out_channels = channels * last if depthwise else last
    accumulators, outputs = [], []
    for b in range(batches):
        for oy in range(oh):
            for ox in range(ow):
                for oc in range(out_channels):
                    total = bias[oc]
                    lane = channel_quant(q, oc)
                    weight_zero = lane.get("weight_zero", w.zero_point)
                    for ky in range(kh):
                        iy = oy * sh + ky * dh - top
                        for kx in range(kw):
                            ix = ox * sw + kx * dw - left
                            for ic in [oc // last] if depthwise else range(channels):
                                av = (
                                    a.data[nhwc_index(a.shape, b, iy, ix, ic)]
                                    if 0 <= iy < height and 0 <= ix < width
                                    else a.zero_point
                                )
                                wi = ((ky * kw + kx) * channels + ic) * last + (
                                    oc % last if depthwise else oc
                                )
                                total += (av - a.zero_point) * (w.data[wi] - weight_zero)
                    integer(total, "accumulator", INT32_MIN, INT32_MAX)
                    accumulators.append(total)
                    outputs.append(requantize(total, lane))
    return accumulators, outputs


def lower_convolution(a, w, geometry, depthwise, channel=None):
    oh, ow, sh, sw, dh, dw, top, left = geometry
    batches, height, width, channels = a.shape
    kh, kw, _, last = w.shape
    m, k, n = batches * oh * ow, kh * kw * (1 if depthwise else channels), last
    if max(m, k, n) > 4096 or m * n * k > 100_000_000 or m * k + k * n > MAX_ELEMENTS:
        raise ValueError("lowered convolution exceeds dense simulator or host element limit")
    data = []
    for b in range(batches):
        for oy in range(oh):
            for ox in range(ow):
                for ky in range(kh):
                    iy = oy * sh + ky * dh - top
                    for kx in range(kw):
                        ix = ox * sw + kx * dw - left
                        for ic in [channel] if depthwise else range(channels):
                            data.append(
                                a.data[nhwc_index(a.shape, b, iy, ix, ic)]
                                if 0 <= iy < height and 0 <= ix < width
                                else a.zero_point
                            )
    weights = (
        w.data
        if not depthwise
        else [
            w.data[((ky * kw + kx) * channels + channel) * last + j]
            for ky in range(kh)
            for kx in range(kw)
            for j in range(last)
        ]
    )
    return m, n, k, data, weights


def lower_depthwise_grouped(a, w, geometry):
    """Depthwise as one grouped GEMM: G = channels, K = kh*kw, N = channel multiplier.
    Row r of the activations holds every channel's window for output position r."""
    oh, ow, sh, sw, dh, dw, top, left = geometry
    batches, height, width, channels = a.shape
    kh, kw, _, last = w.shape
    m, k = batches * oh * ow, kh * kw
    if max(m, k, last) > 4096 or m * k * channels + k * last * channels > MAX_ELEMENTS:
        raise ValueError("lowered convolution exceeds dense simulator or host element limit")
    data = []
    for b in range(batches):
        for oy in range(oh):
            for ox in range(ow):
                for channel in range(channels):
                    for ky in range(kh):
                        iy = oy * sh + ky * dh - top
                        for kx in range(kw):
                            ix = ox * sw + kx * dw - left
                            inside = 0 <= iy < height and 0 <= ix < width
                            data.append(
                                a.data[nhwc_index(a.shape, b, iy, ix, channel)]
                                if inside
                                else a.zero_point
                            )
    weights = [
        w.data[((ky * kw + kx) * channels + channel) * last + j]
        for channel in range(channels)
        for ky in range(kh)
        for kx in range(kw)
        for j in range(last)
    ]
    return m, last, k, data, weights


def grouped_calls(binary):
    """True when npu-sim runs grouped GEMMs (--groups), so depthwise is one call."""
    return sim.supports(binary, "--groups")


def invoke_dense(
    binary,
    work,
    identifier,
    m,
    n,
    k,
    activations,
    weights,
    bias,
    a_zero,
    w_zero,
    q,
    architecture,
    timeout,
    groups=1,
):
    path = work / (str(identifier) + ".npu")
    npufile.write_problem(path, m, n, k, activations, weights, bias, groups)
    flags = {
        "input": path,
        "dump_output": True,
        "input_zero": a_zero,
        "weight_zero": w_zero,
        "output_zero": q["output_zero"],
        "multiplier": q["multiplier"],
        "shift": q["shift"],
        "rounding": q["rounding"],
    }
    if q.get("rescale", "native") != "native":
        if not sim.supports(binary, "--rescale"):
            raise ValueError(
                "this npu-sim build has no --rescale; the TOSA rescale modes need "
                "a build that has it"
            )
        flags["rescale"] = q["rescale"]
    if groups != 1:
        flags["groups"] = groups
    if "channels" in q:
        flags["quantization_file"] = npufile.write_quant(
            work / (str(identifier) + ".quant"), q["channels"]
        )
    flags["relu"] = q["relu"]
    flags.update(architecture)
    result = sim.run(flags, binary, timeout)
    outputs = m * n * groups
    if len(result.get("outputs", [])) != outputs or len(result.get("accumulators", [])) != outputs:
        raise sim.CheckError("simulator did not return complete output arrays")
    return result


def compute_node(node, operands, binary, work, identifier, architecture, timeout):
    a, w = operands[:2]
    require_int8(a, w)
    bias_tensor = operands[2] if len(operands) == 3 else None
    op = node["op"]
    invocations = []
    peak_lowered_bytes = 0
    if op in ("Dense", "MatMul"):
        if len(a.shape) != 2 or len(w.shape) != 2 or a.shape[1] != w.shape[0]:
            raise ValueError("Dense/MatMul requires [M,K] and [K,N]")
        m, k, n = a.shape[0], a.shape[1], w.shape[1]
        if m * n * k > 100_000_000:
            raise ValueError("dense workload exceeds simulator MAC limit")
        output_shape = shape_value([m, n])
        if bias_tensor is not None and bias_tensor.shape != (n,):
            raise ValueError("bias shape must be [N]")
        q = quantization(node, a, w, bias_tensor, n)
        bias = bias_tensor.data if bias_tensor else [0] * n
        expected_acc, expected = dense_reference(a, w, bias, q)
        result = invoke_dense(
            binary,
            work,
            identifier,
            m,
            n,
            k,
            a.data,
            w.data,
            bias,
            a.zero_point,
            w.zero_point,
            q,
            architecture,
            timeout,
        )
        actual, actual_acc = result["outputs"], result["accumulators"]
        invocations.append(result)
    else:
        if len(a.shape) != 4 or len(w.shape) != 4 or a.shape[3] != w.shape[2]:
            raise ValueError("convolution requires NHWC activation and HWIO/HWIM weights")
        depthwise = op == "DepthwiseConv2D"
        channels = a.shape[3]
        out_channels = channels * w.shape[3] if depthwise else w.shape[3]
        geometry = spatial(node, a.shape[1], a.shape[2], w.shape[0], w.shape[1])
        oh, ow = geometry[:2]
        output_shape = shape_value([a.shape[0], oh, ow, out_channels])
        if bias_tensor is not None and bias_tensor.shape != (out_channels,):
            raise ValueError("convolution bias must have one entry per output channel")
        # The MAC limit caps the scalar reference's loop as well as the lowered calls.
        macs = (
            a.shape[0]
            * oh
            * ow
            * out_channels
            * w.shape[0]
            * w.shape[1]
            * (1 if depthwise else channels)
        )
        if macs > 100_000_000:
            raise ValueError("convolution exceeds graph MAC limit")
        lower_m = a.shape[0] * oh * ow
        lower_k = w.shape[0] * w.shape[1] * (1 if depthwise else channels)
        lower_n = w.shape[3]
        if (
            max(lower_m, lower_k, lower_n) > 4096
            or lower_m * lower_k + lower_k * lower_n > MAX_ELEMENTS
        ):
            raise ValueError("lowered convolution exceeds dense simulator or host element limit")
        # Every size limit is checked before the per-channel lanes are built.
        q = quantization(node, a, w, bias_tensor, out_channels)
        bias = bias_tensor.data if bias_tensor else [0] * out_channels
        expected_acc, expected = convolution_reference(a, w, bias, q, geometry, depthwise)
        actual, actual_acc = [0] * len(expected), [0] * len(expected_acc)
        grouped = depthwise and grouped_calls(binary)
        if grouped:
            m, n, k, values, weights = lower_depthwise_grouped(a, w, geometry)
            peak_lowered_bytes = len(values) + len(weights) + len(bias) * 4
            result = invoke_dense(
                binary,
                work,
                identifier,
                m,
                n,
                k,
                values,
                weights,
                bias,
                a.zero_point,
                w.zero_point,
                q,
                architecture,
                timeout,
                channels,
            )
            invocations.append(result)
            actual, actual_acc = list(result["outputs"]), list(result["accumulators"])
        for channel in [] if grouped else range(channels) if depthwise else [None]:
            m, n, k, values, weights = lower_convolution(a, w, geometry, depthwise, channel)
            selected_bias = bias[channel * n : (channel + 1) * n] if depthwise else bias
            peak_lowered_bytes = max(
                peak_lowered_bytes,
                len(values) + (len(weights) if depthwise else 0) + len(selected_bias) * 4,
            )
            invocation_quant = {**q}
            if depthwise and "channels" in q:
                invocation_quant["channels"] = q["channels"][channel * n : (channel + 1) * n]
            result = invoke_dense(
                binary,
                work,
                str(identifier) + "-" + str(channel),
                m,
                n,
                k,
                values,
                weights,
                selected_bias,
                a.zero_point,
                w.zero_point,
                invocation_quant,
                architecture,
                timeout,
            )
            invocations.append(result)
            for row in range(m):
                start = row * out_channels + (channel * n if depthwise else 0)
                actual[start : start + n] = result["outputs"][row * n : (row + 1) * n]
                actual_acc[start : start + n] = result["accumulators"][row * n : (row + 1) * n]
    if actual_acc != expected_acc or actual != expected:
        raise sim.CheckError(
            node["name"] + ": lowered simulator result disagrees with the scalar reference"
        )
    output = Tensor(output_shape, actual, "int8", q["output_scale"], q["output_zero"])
    return output, {
        "execution": "npu",
        "reference_match": True,
        "npu_invocations": len(invocations),
        "fused_relu": q["relu"],
        "cycles": sum(x["cycles"] for x in invocations),
        "macs": sum(x["macs"] for x in invocations),
        "read_bytes": sum(x["read_bytes"] for x in invocations),
        "write_bytes": sum(x["write_bytes"] for x in invocations),
        "peak_lowering_logical_bytes": peak_lowered_bytes,
        "peak_scratchpad_bytes": max(x["scratchpad_bytes"] for x in invocations),
        "invocations": [
            {
                key: x[key]
                for key in (
                    "cycles",
                    "macs",
                    "read_bytes",
                    "write_bytes",
                    "compute_busy_cycles",
                    "dma_busy_cycles",
                    "output_hash",
                    "scratchpad_bytes",
                )
            }
            for x in invocations
        ],
    }


def host_node(node, operands):
    op, a = node["op"], operands[0]
    require_int8(*operands)
    if "quant" in node:
        raise ValueError("host operators preserve quantization; quant overrides are unsupported")
    if op == "Add":
        b = operands[1]
        if a.shape != b.shape or a.scale != b.scale or a.zero_point != b.zero_point:
            raise ValueError(
                "Add requires identical shapes, scales and zero points; no implicit broadcasting"
            )
        data = [saturate(x + y - a.zero_point) for x, y in zip(a.data, b.data, strict=True)]
        shape = a.shape
    elif op == "ReLU":
        data, shape = [max(x, a.zero_point) for x in a.data], a.shape
    elif op == "Flatten":
        if len(a.shape) < 2:
            raise ValueError("Flatten requires rank at least two and preserves batch dimension")
        data, shape = list(a.data), (a.shape[0], math.prod(a.shape[1:]))
        shape_value(list(shape))
    elif op in ("MaxPool2D", "AveragePool2D"):
        if len(a.shape) != 4:
            raise ValueError("pooling requires NHWC")
        kh, kw = pair(node.get("kernel"), "kernel", [2, 2])
        oh, ow, sh, sw, dh, dw, top, left = spatial(node, a.shape[1], a.shape[2], kh, kw)
        include_pad = node.get("count_include_pad", False)
        if not isinstance(include_pad, bool):
            raise ValueError("count_include_pad must be boolean")
        if op == "MaxPool2D" and "count_include_pad" in node:
            raise ValueError("count_include_pad applies only to AveragePool2D")
        shape = shape_value([a.shape[0], oh, ow, a.shape[3]])
        data = []
        for batch in range(a.shape[0]):
            for y in range(oh):
                for x in range(ow):
                    for channel in range(a.shape[3]):
                        window = []
                        for ky in range(kh):
                            iy = y * sh + ky * dh - top
                            for kx in range(kw):
                                ix = x * sw + kx * dw - left
                                if 0 <= iy < a.shape[1] and 0 <= ix < a.shape[2]:
                                    window.append(
                                        a.data[nhwc_index(a.shape, batch, iy, ix, channel)]
                                    )
                                elif op == "AveragePool2D" and include_pad:
                                    window.append(a.zero_point)
                        if not window:
                            raise ValueError("pool window contains no input elements")
                        data.append(
                            max(window)
                            if op == "MaxPool2D"
                            else saturate(
                                round_div(sum(v - a.zero_point for v in window), len(window))
                                + a.zero_point
                            )
                        )
    else:
        raise ValueError("unsupported host operator: " + str(op))
    return Tensor(shape, data, "int8", a.scale, a.zero_point), {
        "execution": "cpu_fallback",
        "cycles": 0,
        "macs": 0,
        "read_bytes": 0,
        "write_bytes": 0,
        "npu_invocations": 0,
        "host_logical_read_bytes": sum(t.bytes for t in operands),
        "host_logical_write_bytes": len(data),
    }


OP_FIELDS = {
    "Dense": {"quant"},
    "MatMul": {"quant"},
    "Conv2D": {"quant", "strides", "dilations", "padding"},
    "DepthwiseConv2D": {"quant", "strides", "dilations", "padding"},
    "Add": set(),
    "ReLU": set(),
    "Flatten": set(),
    "MaxPool2D": {"kernel", "strides", "dilations", "padding"},
    "AveragePool2D": {"kernel", "strides", "dilations", "padding", "count_include_pad"},
}


def validate_graph(document):
    if (
        not isinstance(document, dict)
        or type(document.get("schema_version")) is not int
        or document.get("schema_version") != 1
    ):
        raise ValueError("graph requires schema_version 1")
    if set(document) - {
        "schema_version",
        "name",
        "tensors",
        "inputs",
        "nodes",
        "outputs",
        "expected_outputs",
        "architecture",
        "io_layouts",
    }:
        raise ValueError("unknown graph field")
    if not isinstance(document.get("tensors"), dict) or not document["tensors"]:
        raise ValueError("graph requires an initial tensor map")
    tensors = {}
    for name, spec in document["tensors"].items():
        if not isinstance(name, str) or not name:
            raise ValueError("tensor names must be nonempty strings")
        tensors[name] = parse_tensor(spec)
    inputs = document.get("inputs", [])
    if (
        not isinstance(inputs, list)
        or any(not isinstance(x, str) or x not in tensors for x in inputs)
        or len(set(inputs)) != len(inputs)
    ):
        raise ValueError("inputs must uniquely name initial tensors")
    nodes = document.get("nodes")
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= MAX_NODES:
        raise ValueError("graph must contain one through 256 nodes")
    producers, names = {}, set()
    for i, node in enumerate(nodes):
        if (
            not isinstance(node, dict)
            or not isinstance(node.get("op"), str)
            or node.get("op") not in OP_FIELDS
        ):
            raise ValueError("unsupported or missing operator")
        if set(node) - ({"name", "op", "inputs", "output"} | OP_FIELDS[node["op"]]):
            raise ValueError("unsupported node attributes for " + node["op"])
        if not isinstance(node.get("name"), str) or not node["name"] or node["name"] in names:
            raise ValueError("nodes require unique nonempty names")
        names.add(node["name"])
        output = node.get("output")
        if not isinstance(output, str) or not output or output in tensors or output in producers:
            raise ValueError("node output must have one unique producer")
        producers[output] = i
        operands = node.get("inputs")
        if not isinstance(operands, list) or any(not isinstance(x, str) or not x for x in operands):
            raise ValueError("node inputs must be tensor names")
        count = len(operands)
        if node["op"] in ("Dense", "MatMul", "Conv2D", "DepthwiseConv2D"):
            valid = count in (2, 3)
        else:
            valid = count == (2 if node["op"] == "Add" else 1)
        if not valid:
            raise ValueError("incorrect operator input count")
    successors, indegree = [[] for _ in nodes], [0] * len(nodes)
    for i, node in enumerate(nodes):
        dependencies = set()
        for name in node["inputs"]:
            if name in producers:
                dependencies.add(producers[name])
            elif name not in tensors:
                raise ValueError("missing tensor: " + name)
        indegree[i] = len(dependencies)
        for dependency in dependencies:
            successors[dependency].append(i)
    ready = deque(i for i, degree in enumerate(indegree) if degree == 0)
    order = []
    while ready:
        i = ready.popleft()
        order.append(i)
        for child in successors[i]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(order) != len(nodes):
        raise ValueError("graph has a dependency cycle")
    outputs = document.get("outputs")
    if (
        not isinstance(outputs, list)
        or not outputs
        or any(not isinstance(x, str) or x not in producers and x not in tensors for x in outputs)
        or len(set(outputs)) != len(outputs)
    ):
        raise ValueError("outputs must uniquely name graph tensors")
    layouts = document.get("io_layouts", {})
    if not isinstance(layouts, dict) or set(layouts) - set(outputs):
        raise ValueError("io_layouts must name declared graph outputs")
    for layout in layouts.values():
        if layout != {"internal": "NHWC", "external": "NCHW"}:
            raise ValueError("unsupported external I/O layout conversion")
    golden = document.get("expected_outputs", {})
    if not isinstance(golden, dict) or set(golden) - set(outputs):
        raise ValueError("expected_outputs must name declared outputs")
    for values in golden.values():
        if not isinstance(values, list):
            raise ValueError("expected output must be a flat array")
        for value in values:
            integer(value, "expected output", -128, 127)
    return tensors, nodes, order, outputs, golden


def external_output(tensor, layout):
    if layout is None:
        return tensor.descriptor(include_data=True)
    if len(tensor.shape) != 4:
        raise ValueError("NHWC to NCHW output conversion requires rank four")
    batches, height, width, channels = tensor.shape
    data = [
        tensor.data[nhwc_index(tensor.shape, b, y, x, c)]
        for b in range(batches)
        for c in range(channels)
        for y in range(height)
        for x in range(width)
    ]
    result = tensor.descriptor()
    result.update(shape=[batches, channels, height, width], data=data)
    return result


def run_graph(document, binary, architecture=None, timeout=60):
    tensors, nodes, order, outputs, golden = validate_graph(document)
    if not isinstance(document.get("architecture", {}), dict) or (
        architecture is not None and not isinstance(architecture, dict)
    ):
        raise ValueError("architecture must be an object")
    architecture = {**document.get("architecture", {}), **(architecture or {})}
    for key, value in architecture.items():
        if not isinstance(key, str) or not key.replace("_", "").isalnum() or key in RUNTIME_FLAGS:
            raise ValueError(f"unsupported architecture field: {key}")
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ValueError(f"architecture field {key} must be a number or a string")
    architecture.setdefault("timing", "event")
    uses = Counter(name for node in nodes for name in node["inputs"])
    for name in outputs:
        uses[name] += 1
    live = {name: tensor for name, tensor in tensors.items() if uses[name]}
    parameter_names = {
        name
        for node in nodes
        if node["op"] in ("Dense", "MatMul", "Conv2D", "DepthwiseConv2D")
        for name in node["inputs"][1:]
    }
    activation_names = set(tensors) - parameter_names
    activation_names.update(document.get("inputs", []))
    activation_names.update(node["output"] for node in nodes)
    activation_names.update(node["inputs"][0] for node in nodes)

    def activation_bytes():
        return sum(tensor.bytes for name, tensor in live.items() if name in activation_names)

    peak_live_bytes = sum(t.bytes for t in live.values())
    peak_live_activation_bytes = activation_bytes()
    initial_bytes = peak_live_bytes
    tensor_metadata = {name: t.descriptor() for name, t in tensors.items()}
    tensors.clear()
    layers, freed = [], []
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="npu-graph-") as directory:
        work = Path(directory)
        for position, index in enumerate(order):
            node = nodes[index]
            operands = [live[name] for name in node["inputs"]]
            if node["op"] in ("Dense", "MatMul", "Conv2D", "DepthwiseConv2D"):
                output, details = compute_node(
                    node, operands, binary, work, position, architecture, timeout
                )
            else:
                output, details = host_node(node, operands)
            live[node["output"]] = output
            tensor_metadata[node["output"]] = output.descriptor()
            peak_live_bytes = max(peak_live_bytes, sum(t.bytes for t in live.values()))
            peak_live_activation_bytes = max(peak_live_activation_bytes, activation_bytes())
            released = []
            for name in node["inputs"]:
                uses[name] -= 1
                if uses[name] == 0:
                    del live[name]
                    released.append(name)
            if uses[node["output"]] == 0:
                del live[node["output"]]
                released.append(node["output"])
            freed += released
            layers.append(
                {
                    "name": node["name"],
                    "operator": node["op"],
                    "output": node["output"],
                    "output_shape": list(output.shape),
                    "released_tensors": released,
                    "live_tensor_bytes_after": sum(t.bytes for t in live.values()),
                    "live_activation_bytes_after": activation_bytes(),
                    **details,
                }
            )
    for name, expected in golden.items():
        if live[name].data != expected:
            raise sim.CheckError("graph output differs from expected_outputs: " + name)
    return {
        "schema_version": 1,
        "workload": document.get("name", "graph"),
        "outputs": {name: live[name].descriptor(include_data=True) for name in outputs},
        "external_outputs": {
            name: external_output(live[name], document.get("io_layouts", {}).get(name))
            for name in outputs
        },
        "golden_outputs_checked": sorted(golden),
        "layers": layers,
        "tensor_metadata": tensor_metadata,
        "sum_npu_cycles": sum(x["cycles"] for x in layers),
        "macs": sum(x["macs"] for x in layers),
        "npu_read_bytes": sum(x["read_bytes"] for x in layers),
        "npu_write_bytes": sum(x["write_bytes"] for x in layers),
        "cpu_fallback_nodes": [x["name"] for x in layers if x["execution"] == "cpu_fallback"],
        "initial_live_tensor_bytes": initial_bytes,
        "peak_live_tensor_bytes": peak_live_bytes,
        "peak_live_activation_bytes": peak_live_activation_bytes,
        "peak_lowering_logical_bytes": max(x.get("peak_lowering_logical_bytes", 0) for x in layers),
        "final_live_tensor_bytes": sum(t.bytes for t in live.values()),
        "released_tensors": freed,
        "host_elapsed_seconds": time.perf_counter() - started,
        "layer_overlap": False,
        "sram_residency": False,
        "host_ops_in_npu_cycles": False,
    }


def load_graph(path):
    return jsonio.load_strict(path, max_bytes=32 * 1024 * 1024)


def layer_table(report):
    rows = [
        {
            "layer": x["name"],
            "op": x["operator"],
            "where": x["execution"],
            "calls": x["npu_invocations"],
            "macs": x["macs"],
            "cycles": x["cycles"],
            "dram bytes": x["read_bytes"] + x["write_bytes"],
        }
        for x in report["layers"]
    ]
    return table.format_table(rows) + (
        f"\n{report['sum_npu_cycles']:,} NPU cycles, "
        f"{report['macs']:,} MACs; outputs match the reference\n"
    )


@cli.main_guard("graph_runtime")
def main(argv: Sequence[str] | None = None) -> int:
    parser = cli.Parser("graph_runtime", __doc__)
    parser.add_argument("graph", type=Path, help="graph JSON (see examples/)")
    cli.add_binary_options(parser)
    cli.add_format_option(parser, default="json")
    parser.add_argument(
        "--timing",
        choices=["event", "cycle"],
        help="timing mode of every call; overrides the graph's architecture",
    )
    args = parser.parse_args(argv)
    override = {"timing": args.timing} if args.timing else None
    report = run_graph(load_graph(args.graph), cli.simulator(args), override, args.timeout)
    print(jsonio.dumps(report) if args.format == "json" else layer_table(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
