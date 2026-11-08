"""Run npu-sim and its sibling executables.

``run()`` takes an argv-style flag list or a dict of flags and passes every option through
unchecked, so a new npu-sim option needs no change here: ``{"tile_m": 8, "relu": True}``
becomes ``--tile-m 8 --relu``. ``LiveSession`` drives the ``--live-stdio`` protocol.
"""

from __future__ import annotations

import collections
import contextlib
import functools
import json
import math
import queue
import subprocess
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import binary as binaries

Flags = Mapping[str, Any] | Sequence[Any]
MAX_LIVE_MESSAGE = 1_048_576
# Result and snapshot layouts this package reads. Version 2 nests "config" by option-table
# path (shape.m, arch.frequency_mhz, ...) and adds "memory", "cluster" and "cores"; version 1
# carried a flat config. config_value() reads either layout.
RESULT_SCHEMA_VERSIONS = (1, 2)
_MISSING = object()


class SimError(RuntimeError):
    """An executable exited with an error or produced unusable output."""

    def __init__(
        self,
        message: str,
        returncode: int | None = None,
        stderr: str = "",
        command: Sequence[str] = (),
    ):
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr
        self.command = list(command)


class CheckError(RuntimeError):
    """A simulator result failed an independent check."""


class LiveError(RuntimeError):
    """The --live-stdio child misbehaved or reported an error."""


def format_value(value: Any) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"flag value must be finite: {value}")
        return format(value, ".17g")
    return str(value)


def flag_name(key: str) -> str:
    return key if key.startswith("--") else "--" + key.replace("_", "-")


def to_flags(config: Flags) -> list[str]:
    """Flag list for ``config``: a dict (True = bare switch, False/None = omitted) or argv."""
    if isinstance(config, Mapping):
        flags: list[str] = []
        for key, value in config.items():
            if value is None or value is False:
                continue
            flags.append(flag_name(str(key)))
            if value is True:
                continue
            if isinstance(value, (list, tuple)):
                flags.append(",".join(format_value(item) for item in value))
            else:
                flags.append(format_value(value))
        return flags
    if isinstance(config, (str, bytes)):
        raise TypeError("flags must be a list of tokens or a dict, not a string")
    return [format_value(token) for token in config]


def _scalar(text: str) -> Any:
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def from_flags(tokens: Sequence[Any]) -> dict[str, Any]:
    """Inverse of ``to_flags`` for argv lists:
    ``--tile-m 8 --relu`` -> {"tile_m": 8, "relu": True}."""
    tokens = [str(token) for token in tokens]
    result: dict[str, Any] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            raise ValueError(f"expected a --flag, got {token!r}")
        key = token[2:].replace("-", "_")
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            result[key] = _scalar(tokens[index + 1])
            index += 2
        else:
            result[key] = True
            index += 1
    return result


# Flat result-config keys (npu-sim flag names) and the option-table paths (src/options.cpp)
# that hold them in a nested config. Several leaves are renamed (weight_bits ->
# weights.bits), so a leaf-name search alone misses them.
CONFIG_PATHS = {
    "m": "shape.m",
    "n": "shape.n",
    "k": "shape.k",
    "groups": "shape.groups",
    "seed": "shape.seed",
    "macs_per_cycle": "arch.compute.macs_per_cycle",
    "array_rows": "arch.compute.array_rows",
    "array_cols": "arch.compute.array_cols",
    "dataflow": "arch.compute.dataflow",
    "pipeline_cycles": "arch.compute.pipeline_cycles",
    "weight_decode_values_per_cycle": "arch.compute.weight_decode_values_per_cycle",
    "memory_model": "arch.memory.model",
    "memory_bytes_per_cycle": "arch.memory.bytes_per_cycle",
    "memory_latency": "arch.memory.latency",
    "dma_burst_bytes": "arch.memory.burst_bytes",
    "dram_bytes_per_cycle": "arch.memory.dram_bytes_per_cycle",
    "axi_data_bytes": "arch.memory.axi.data_bytes",
    "axi_max_burst_beats": "arch.memory.axi.max_burst_beats",
    "axi_read_outstanding": "arch.memory.axi.read_outstanding",
    "axi_write_outstanding": "arch.memory.axi.write_outstanding",
    "axi_read_latency": "arch.memory.axi.read_latency",
    "axi_write_latency": "arch.memory.axi.write_latency",
    "sram_bytes": "arch.sram_bytes",
    "weight_cache_bytes": "arch.weight_cache_bytes",
    "frequency_mhz": "arch.frequency_mhz",
    "cores": "arch.cores",
    "tile_m": "mapping.tile_m",
    "tile_n": "mapping.tile_n",
    "tile_k": "mapping.tile_k",
    "buffer_slots": "mapping.buffer_slots",
    "panel_buffers": "mapping.panel_buffers",
    "partition": "mapping.partition",
    "weight_bits": "weights.bits",
    "weight_encoding": "weights.encoding",
    "palette_bits": "weights.palette_bits",
    "sparse_block": "weights.sparse_block",
    "timing": "sim.timing",
    "timing_only": "sim.timing_only",
    "depthwise_macs_per_cycle": "arch.compute.depthwise_macs_per_cycle",
    "activation_cache_bytes": "arch.activation_cache_bytes",
    **{
        f"conv_{leaf}": f"shape.conv.{leaf}"
        for leaf in (
            "input_h",
            "input_w",
            "input_c",
            "kernel_h",
            "kernel_w",
            "stride_h",
            "stride_w",
            "dilation_h",
            "dilation_w",
            "pad_top",
            "pad_left",
            "pad_bottom",
            "pad_right",
        )
    },
}
FLAT_KEYS = {path: key for key, path in CONFIG_PATHS.items()}


def _at_path(config: Mapping[str, Any], path: str) -> Any:
    node: Any = config
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return _MISSING
        node = node[part]
    return node


def config_value(config: Mapping[str, Any], key: str, default: Any = _MISSING) -> Any:
    """Look up ``key`` in a result's config, flat (schema_version 1) or nested by option-table
    path (schema_version 2).

    ``key`` is a flat key ("weight_bits"), an option-table path ("weights.bits") or a leaf
    name. Known keys resolve through CONFIG_PATHS in both directions; anything else falls
    back to a unique nested leaf of that name. Missing keys raise KeyError unless a
    ``default`` is given."""
    flat = FLAT_KEYS.get(key, key)
    path = CONFIG_PATHS.get(flat, key)
    for candidate in (path, flat):
        if candidate in config:
            return config[candidate]
        value = _at_path(config, candidate) if "." in candidate else _MISSING
        if value is not _MISSING:
            return value
    name = flat.rsplit(".", 1)[-1]
    found = []

    def visit(node: Mapping[str, Any]) -> None:
        for child, value in node.items():
            if child == name and not isinstance(value, Mapping):
                found.append(value)
            elif isinstance(value, Mapping):
                visit(value)

    visit(config)
    if len(found) == 1:
        return found[0]
    if default is _MISSING:
        raise KeyError(key)
    return default


def _nonnegative_int(result: Mapping[str, Any], key: str) -> int:
    value = result.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CheckError(f"invalid nonnegative integer result: {key}")
    return value


def result_numerics(result: Mapping[str, Any]) -> str | None:
    """ "bit-exact" (numerical_match true), "skipped" (a --timing-only run, which reports
    numerical_match null or omits it), or None for any other combination. A result without
    a "numerics" field is bit-exact when numerical_match is true."""
    match = result.get("numerical_match")
    numerics = result.get("numerics", "bit-exact" if match is True else None)
    if numerics == "bit-exact" and match is True:
        return numerics
    if numerics == "skipped" and match is None:
        return numerics
    return None


def schema_version(value: Any) -> int | None:
    """The result or snapshot layout version of ``value``; None for a layout outside
    RESULT_SCHEMA_VERSIONS."""
    if not isinstance(value, Mapping):
        return None
    version = value.get("schema_version")
    if type(version) is not int or version not in RESULT_SCHEMA_VERSIONS:
        return None
    return version


def validate_result(result: Any) -> dict:
    """Reject anything that is not a complete, self-consistent npu-sim result."""
    if not isinstance(result, dict) or schema_version(result) is None:
        raise CheckError("expected a JSON object with schema_version 1 or 2")
    numerics = result_numerics(result)
    if numerics is None:
        raise CheckError("simulator did not report full numerical agreement")
    for key in (
        "cycles",
        "macs",
        "read_bytes",
        "write_bytes",
        "compute_busy_cycles",
        "dma_busy_cycles",
        "peak_inflight_tiles",
        "slot_wait_cycles",
        "scratchpad_bytes",
    ):
        _nonnegative_int(result, key)
    if result["cycles"] == 0:
        raise CheckError("cycle count must be positive")
    clustered = isinstance(result.get("cores"), list) and len(result["cores"]) > 1
    if not clustered:
        for key in ("compute_busy_cycles", "dma_busy_cycles"):
            if result[key] > result["cycles"]:
                raise CheckError(f"resource busy cycles exceed elapsed cycles: {key}")
    for key in (
        "simulation_wall_seconds",
        "compute_utilization",
        "memory_utilization",
        "modeled_latency_ns",
    ):
        value = result.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise CheckError(f"invalid finite nonnegative result: {key}")
        if key.endswith("utilization") and value > 1 and not clustered:
            raise CheckError(f"utilization must be a fraction: {key}")
    if not isinstance(result.get("config"), dict):
        raise CheckError("missing configuration object")
    output_hash = result.get("output_hash")
    if numerics == "bit-exact" and (
        isinstance(output_hash, bool) or not isinstance(output_hash, (str, int))
    ):
        raise CheckError("missing deterministic output hash")
    return result


@dataclass(eq=False)
class SimResult(Mapping):
    """One npu-sim result; reads like the JSON dict it wraps."""

    raw: dict
    command: list[str] = field(default_factory=list)
    wall_seconds: float = 0.0

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.raw)

    def __len__(self) -> int:
        return len(self.raw)

    @property
    def cycles(self) -> int:
        return self.raw["cycles"]

    @property
    def macs(self) -> int:
        return self.raw["macs"]

    @property
    def dram_bytes(self) -> int:
        return self.raw["read_bytes"] + self.raw["write_bytes"]

    @property
    def config(self) -> dict:
        return self.raw["config"]

    @property
    def numerics(self) -> str:
        return self.raw.get("numerics", "bit-exact")

    def as_run(self) -> dict:
        return {
            "command": self.command,
            "subprocess_wall_seconds": self.wall_seconds,
            "result": self.raw,
        }


def execute(binary: Path | str, flags: Flags, timeout: float = 60) -> tuple[dict, list[str], float]:
    """Run an executable that prints one JSON object; return (object, argv, wall seconds)."""
    command = [str(binary)] + to_flags(flags)
    started = time.perf_counter()
    try:
        process = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except OSError as error:
        raise SimError(f"cannot run {binary}: {error}", command=command) from error
    elapsed = time.perf_counter() - started
    name = Path(str(binary)).name
    if process.returncode != 0:
        detail = " ".join(process.stderr.split())[:2000]
        raise SimError(
            f"{name} exited with status {process.returncode}: {detail}",
            process.returncode,
            process.stderr,
            command,
        )
    try:
        value = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise SimError(
            f"{name} stdout is not one JSON object", 0, process.stderr, command
        ) from error
    if not isinstance(value, dict) or schema_version(value) is None:
        raise SimError(
            f"{name} did not return a schema_version 1 or 2 object", 0, process.stderr, command
        )
    return value, command, elapsed


def run(
    config: Flags,
    binary: Path | str | None = None,
    timeout: float = 60,
    *,
    check: bool = True,
    build_dir: Path | str | None = None,
) -> SimResult:
    """Run npu-sim once. ``check`` validates the result (counters, numerics, hash)."""
    executable = binaries.resolve(binary, "npu-sim", build_dir)
    value, command, elapsed = execute(executable, config, timeout)
    if check:
        validate_result(value)
    return SimResult(value, command, elapsed)


@functools.lru_cache(maxsize=16)
def _help(binary: str, mtime: float) -> str:
    try:
        process = subprocess.run(
            [binary, "--help"], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return process.stdout + process.stderr


def help_text(binary: Path | str) -> str:
    path = Path(binary)
    return _help(str(path), path.stat().st_mtime if path.exists() else 0.0)


def supports(binary: Path | str, flag: str, value: str | None = None) -> bool:
    """True when ``binary --help`` documents ``flag`` (and ``value`` on the same line)."""
    flag = flag_name(flag)
    for line in help_text(binary).splitlines():
        words = line.replace(",", " ").replace("|", " ").split()
        if flag in words and (value is None or value in words or value in line):
            return True
    return False


class LiveSession:
    """One ``npu-sim --live-stdio`` child: send a command line, read one JSON reply."""

    def __init__(self, binary: Path | str, flags: Flags = (), response_timeout: float = 10.0):
        self.binary = Path(binary)
        self.response_timeout = response_timeout
        self.messages: queue.Queue = queue.Queue(maxsize=8)
        self.stderr_tail: collections.deque[str] = collections.deque(maxlen=20)
        self.last: dict = {}
        command = [str(self.binary), "--live-stdio", *to_flags(flags)]
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except OSError as error:
            raise LiveError(f"cannot start {self.binary.name}: {error}") from error
        self._readers = [
            threading.Thread(target=self._read_stdout, name="npu-live-stdout", daemon=True),
            threading.Thread(target=self._read_stderr, name="npu-live-stderr", daemon=True),
        ]
        for reader in self._readers:
            reader.start()

    @property
    def args(self) -> list[str]:
        return list(self.process.args)

    def _read_stdout(self) -> None:
        stdout = self.process.stdout
        assert stdout is not None
        try:
            while True:
                line = stdout.readline(MAX_LIVE_MESSAGE + 1)
                if not line:
                    self.messages.put(LiveError("simulator exited before replying"), timeout=1)
                    return
                if len(line) > MAX_LIVE_MESSAGE:
                    raise LiveError("simulator reply exceeds 1 MiB")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise LiveError("simulator reply is not a JSON object")
                self.messages.put(value, timeout=self.response_timeout)
        except (ValueError, OSError, LiveError, queue.Full) as error:
            with contextlib.suppress(queue.Full):
                self.messages.put_nowait(LiveError(f"invalid simulator reply: {error}"))

    def _read_stderr(self) -> None:
        stderr = self.process.stderr
        assert stderr is not None
        try:
            for line in stderr:
                self.stderr_tail.append(line.rstrip())
        except (OSError, ValueError):
            pass

    def receive(self) -> dict:
        try:
            value = self.messages.get(timeout=self.response_timeout)
        except queue.Empty as error:
            raise LiveError(
                f"simulator did not reply within {self.response_timeout:g} s"
            ) from error
        if isinstance(value, Exception):
            if self.stderr_tail:
                raise LiveError(f"{value}: {self.stderr_tail[-1]}")
            raise value
        if "error" in value:
            raise LiveError(str(value["error"])[:300])
        self.last = value
        return value

    def command(self, text: str) -> dict:
        stdin = self.process.stdin
        if stdin is None or self.process.poll() is not None:
            raise LiveError("simulator is not running")
        try:
            stdin.write(text + "\n")
            stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise LiveError("simulator input closed") from error
        return self.receive()

    def advance(self, cycles: int) -> dict:
        return self.command(f"advance {int(cycles)}")

    def snapshot(self) -> dict:
        return self.command("snapshot")

    def close(self) -> None:
        """Ask the child to stop; terminate it if it does not exit promptly."""
        process = self.process
        try:
            if process.poll() is None:
                try:
                    assert process.stdin is not None
                    process.stdin.write("stop\n")
                    process.stdin.flush()
                    process.wait(timeout=0.5)
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    process.terminate()
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=1)
        finally:
            self._readers[1].join(timeout=1)  # stderr reaches EOF once the child exits
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    with contextlib.suppress(OSError):
                        stream.close()

    def __enter__(self) -> LiveSession:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
