# Coefficient-based energy estimates

`scripts/energy_report.py` adds a small, inspectable energy model to the architecture simulator. It consumes a successful `npu-sim` JSON result and requires explicit MAC and external-memory coefficients. There are no invented default silicon energy numbers.

```sh
build/npu-sim --m 16 --n 16 --k 32 > results/example.json
python3 scripts/energy_report.py --result results/example.json \
  --mac-pj "$MAC_PJ" --dram-byte-pj "$DRAM_BYTE_PJ" \
  --static-mw "$STATIC_MW" --output results/example-energy.json
```

Create the results directory first. Set the coefficient variables from your own measurements, technology model or explicitly declared experimental assumptions. `--static-mw` is optional and defaults to zero, excluding the static component. The optional output path must be new; the tool never overwrites an existing result. JSON is also written to stdout.

## Counts, coefficients and units

The calculation is

$$
\begin{aligned}
E_{\mathrm{MAC,pJ}}&=N_{\mathrm{MAC}}e_{\mathrm{MAC,pJ}},\\
E_{\mathrm{read,pJ}}&=B_{\mathrm{read}}e_{\mathrm{DRAM,pJ/byte}},\\
E_{\mathrm{write,pJ}}&=B_{\mathrm{write}}e_{\mathrm{DRAM,pJ/byte}},\\
E_{\mathrm{static,pJ}}&=t_{\mathrm{ns}}P_{\mathrm{static,mW}},\\
E_{\mathrm{total,pJ}}&=E_{\mathrm{MAC,pJ}}+E_{\mathrm{read,pJ}}+E_{\mathrm{write,pJ}}+E_{\mathrm{static,pJ}},\\
P_{\mathrm{average,mW}}&=\frac{E_{\mathrm{total,pJ}}}{t_{\mathrm{ns}}}.
\end{aligned}
$$

One milliwatt applied for one nanosecond contributes one picojoule. Nanojoules are reported by dividing picojoules by one thousand. Reads and writes currently share the same coefficient; a future asymmetric memory model would require separate coefficients and tests.

The report includes component energies, total energy in both units, modeled latency, average estimated power, source counts/hash, the exact coefficients and limitations. The latency must agree with simulator cycles and configured frequency. Only schema-version-one simulator results with full numerical agreement are accepted. Boolean counts, negative or nonfinite values, numeric overflow, malformed JSON and duplicate keys are rejected. A graph-runtime report is a different schema contract and is rejected.

## Interpretation and calibration

Action counts combined with component costs are an established architecture-estimation approach; see the primary [Timeloop/Accelergy energy-reference-table documentation](https://timeloop.csail.mit.edu/v4/parsing_and_intermediate_files/energy-and-area-reference-tables). This script is an independent, much smaller coefficient model and does not invoke Accelergy or include its component models.

MACs and bytes come from this simulator's scheduling and traffic contract. Cache hits or INT4 packing can reduce counted external-memory bytes. No sparse execution, unreported data movement or omitted subsystem receives an automatic energy benefit.

The estimate excludes SRAM/register accesses, clock trees, interconnect, DMA control, decompression, host work and process-launch cost. The constant static-power term has no voltage, temperature, power-state or thermal feedback model. A coefficient calibrated for one memory system or voltage is not automatically transferable to another configuration. Coefficients should state whether their measured cost already includes another component, to avoid double counting.

Energy differences support comparisons under fixed, declared assumptions. They do not establish measured device power, battery life, thermal safety or silicon energy accuracy. Hardware calibration is a separate validation step.

## Verification

```sh
python3 tests/energy_tests.py
```

Tests use hand-calculated unit examples, zero-cost components, schema and numerical-agreement checks, invalid/overflowing values, latency consistency, required coefficient flags, strict JSON and create-only output behavior. Their arithmetic coefficients are synthetic fixtures, not hardware measurements.
