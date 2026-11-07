"""Python tooling for the npu-sim SystemC model.

Standard library only. Every tool runs from a checkout with ``python -m npuperf.<tool>``:

    binary         locate the CMake-built simulators
    sim            run npu-sim from flags or a dict, drive its --live-stdio protocol
    analytic       closed-form traffic, busy-cycle and bound model of the simulator
    roofline       roofline points and plots
    workloads      canonical network layer tables lowered to (grouped) GEMMs
    svg            dependency-free SVG plotting
    quant          integer requantisation oracle (native, TOSA single and double rounding)
    graph          quantised graph runtime on the accelerator model
    sweep, autotune, compare_modes, energy, mobile_suite, live    experiment tools
"""
