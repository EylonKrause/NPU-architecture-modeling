# Integer arithmetic and quantization

The numerical datapath implements signed INT8 activations and either signed INT8 or signed INT4 weights. Bias is INT32. INT4 weights remain sign-extended INT8 values in the host tensor file and arithmetic implementation; the architecture traffic model packs two weights per byte. This separates simulated storage from the portable C++ representation.

## Per-output-channel parameters

Each output column may have its own weight zero point, integer multiplier, and right shift. An omitted vector uses the corresponding scalar parameter for every column. Each provided vector must contain exactly `N` values. The `--quantization-file` CLI option accepts all three vectors as a small text file. Its header is `NPUQ1 N`, followed by exactly `N` rows of `weight_zero multiplier shift`, without labels. For example, save this as `channels.npuq`:

```text
NPUQ1 3
1 1 1
-2 3 2
3 2 0
```

```sh
./build/npu-sim --m 2 --n 3 --k 2 --weight-bits 4 \
  --quantization-file channels.npuq \
  --input-zero 1 --output-zero 5 --dump-output
```

With input zero point $z_a$, output-channel weight zero point $z_{w,c}$, and bias $b_c$, the complete dot product is

$$a_{r,c}=b_c+\sum_j(A_{r,j}-z_a)(W_{j,c}-z_{w,c}).$$

The implementation uses an INT64 temporary and rejects a final accumulator outside the INT32 range. Intermediate values may cancel; there is no intermediate saturation. This is an explicit architectural arithmetic choice, not a claim that every hardware accumulator behaves this way.

For nonnegative integer multiplier $M_c$ and shift $s_c$, requantization rounds the magnitude of $a_{r,c}M_c/2^{s_c}$ to nearest. The default `--rounding away` breaks ties away from zero; `--rounding even` breaks ties toward the nearest even integer. For example, nearest-even maps positive one-half to zero, positive three-halves to two, and positive five-halves to two, with symmetric negative results. It then adds the output zero point, applies optional ReLU at that zero point, and saturates to signed INT8. Shifts range from zero through 31, and multipliers from zero through `INT32_MAX`. The maximum product fits INT64, including `INT32_MIN` multiplied by `INT32_MAX`.

Weight zero points must be representable by the selected weight width: signed INT4 uses the range from negative eight through seven. Activations and output zero points retain the signed INT8 range. Tensor file weights outside the selected range are rejected; no silent bit truncation or wraparound occurs.

Per-channel lookup uses the global output column even when a tile begins at a nonzero column. Quantization does not depend on tile shape, reduction chunk size, or timing mode. The scalar `requantize(accumulator, quant)` C++ overload uses channel zero; callers applying channel vectors should use the explicit channel overload.

## Verification and interoperability

C++ tests include manually calculated asymmetric-channel examples, negative half ties in both rounding modes, zero multipliers, extreme signed products, final accumulator overflow, ReLU, and nonzero tile columns. An independent Python test computes expected results with unbounded integers from external tensor files, exercising random asymmetric parameters and INT4 endpoints.

The optional ONNX adapter selects nearest-even explicitly; the default CLI retains the original nearest-away behavior.

This arithmetic contract is intentionally explicit. Framework conversion must account for rounding rules, multiplier conventions, zero-point restrictions, tensor layout, and bias scales. Loading quantized integers does not make the model automatically bit compatible with a particular framework or a proprietary NPU.
