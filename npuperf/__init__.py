"""Python tooling for the npu-sim SystemC model.

Standard library only, except npuperf.onnx_import (numpy, onnx). Every tool runs from a
checkout with ``python -m npuperf.<tool>``:

    binary         locate the CMake-built simulators
    sim            run npu-sim from flags or a dict, drive its --live-stdio protocol
    analytic       closed-form traffic, busy-cycle and bound model, and a request-level replay
    mapper         per-layer tile, slot, panel-depth and cache-split search
    roofline       roofline points and plots
    designs        the configs/ design points, their headline figures and sizing rules
    workloads      canonical network layer tables lowered to (grouped) GEMMs
    studies        the studies of docs/results.md; reproduce regenerates its tables and figures
    svg            dependency-free SVG plotting
    quant          integer requantization oracle (native, TOSA single and double rounding)
    graph          quantized graph runtime on the accelerator model
    graph_optimize ReLU fusion pass over graph JSON
    onnx_import    QLinearMatMul/QLinearConv ONNX models to graph JSON
    sweep, autotune, compare_modes, energy, mobile_suite, mlp_demo, live    experiment tools
"""
