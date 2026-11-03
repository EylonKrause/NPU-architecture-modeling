# Run the live NPU simulator

After installing the one-time build prerequisites below, start everything with
one Python command. No `pip` packages, API key, cloud account, Apple chip or
physical NPU hardware is required. The Python launcher builds and runs the
project's native C++/SystemC simulator on your computer.

## Start

Clone the repository, or download and extract its ZIP from GitHub:

```sh
git clone https://github.com/EylonKrause/NPU-architecture-modeling.git
cd NPU-architecture-modeling
```

On Linux, macOS or WSL:

```sh
python3 run.py
```

On native Windows, in PowerShell or Command Prompt:

```powershell
py -3 run.py
```

The launcher configures and compiles the simulator, starts continuous validated
inference, and opens your default browser at **http://127.0.0.1:8765**. Leave the
terminal running. Use **Ctrl+C** in that terminal to stop the server and its
simulator process. Pause, step and resume are available in the browser.

The first run needs internet access to download hash-pinned SystemC and can take
several minutes to compile. Builds use the dedicated `build-live` folder by
default. Later runs reuse that build and compile only what changed. After the
dependency is cached, the live dashboard itself needs no internet connection.

This is an executable architectural simulator. Each completed inference checks
every output against a reference; the browser reports real SystemC execution.
Its deliberately paced display is not a measurement of physical NPU throughput.

## One-time prerequisites

Install **Python 3.10 or newer**, **CMake 3.24 or newer**, a **C++17 compiler**,
and its native build tools. Python starts the build automatically, but cannot
compile C++ without that toolchain. The launcher does not install system
packages or change your machine's configuration.

### Ubuntu / Debian / WSL

In an Ubuntu or Debian terminal:

```sh
sudo apt-get update
sudo apt-get install build-essential cmake ninja-build python3
cmake --version
python3 --version
```

Check that the reported versions meet the minimums above. If your distribution
ships an older CMake, use a current stable release from the
[official CMake downloads](https://cmake.org/download/).

For WSL, run both the build and launcher inside the Linux terminal. Open the
printed local URL in your Windows browser if automatic browser opening is not
available.

### macOS

Install Apple's compiler and command-line build tools:

```sh
xcode-select --install
```

Finish the installation dialog. Install [Python](https://www.python.org/downloads/)
and the stable macOS package from [CMake](https://cmake.org/download/) if needed.
Make sure `python3` and `cmake` are available in a new terminal. The CMake app
provides command-line installation instructions under **Tools**. Apple's
[command-line tools guide](https://developer.apple.com/documentation/xcode/installing-the-command-line-tools/)
explains the compiler setup.

### Native Windows

1. Install [Python](https://www.python.org/downloads/windows/) and verify
   `py -3 --version` in a new terminal.
2. Install a stable [CMake](https://cmake.org/download/) release and enable its
   option to add CMake to `PATH`.
3. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)
   with the **Desktop development with C++** workload, including the compiler
   and Windows SDK. A full Visual Studio IDE is not required; see Microsoft's
   [Build Tools installation guide](https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-170).
4. Open a new terminal in the repository and run `py -3 run.py`.

The launcher handles the native `.exe` name and Release build directory.
The [automated checks](https://github.com/EylonKrause/NPU-architecture-modeling/actions/workflows/ci.yml)
include native Windows and macOS builds, build reuse and live launcher checks,
alongside the Linux regression suites.
For current MSVC compilers, the fetched SystemC dependency receives the
[official upstream integer-stream compatibility fix](https://github.com/accellera-official/systemc/commit/3570c374f3c98e060d6267183ccb8de51aab5129).
The build checks newline-normalized original and patched content hashes,
preserves dependency copyrights and refuses to overwrite unexpected local
modifications.

## Useful options

Use `py -3` instead of `python3` for these examples on native Windows.

```sh
# Keep the browser closed; print the URL for opening manually.
python3 run.py --no-browser

# Use another local port if 8765 is already occupied.
python3 run.py --port 8766

# Let the operating system choose an available local port.
python3 run.py --port 0

# Start with actual packed palette weights and their decoding costs.
python3 run.py --profile palette-weights

# Select a separate build directory; quote paths that contain spaces.
python3 run.py --build-dir "build live"

# Use an existing compiled build without configuring or invoking CMake.
python3 run.py --skip-build --build-dir build

# Limit or increase parallel compilation (default: 2; allowed: 1 through 128).
python3 run.py --jobs 4

# Compile without starting the dashboard.
python3 run.py --build-only

python3 run.py --help
```

`--build-only` and `--skip-build` cannot be combined. Both modes run a numerical
startup check. `--skip-build` requires a compatible simulator binary in the
selected build directory and does not rebuild changes made since compilation.

The server listens only on `127.0.0.1`, so the URL is local to the computer
running it. Changing the port does not expose it to the LAN. If a port is busy,
the launcher reports the conflict before building and suggests `--port 8766`;
choose another port or stop the existing instance yourself. It does not
terminate someone else's server. With `--port 0`, use the actual URL printed
by the running server.

Run the command from the extracted repository, or quote the full script path
when the folder contains spaces:

```sh
python3 "/path/to/NPU architecture modeling/run.py" --no-browser
```

For headless machines or environments without a default browser, use
`--no-browser`. If a build fails, read the compiler/CMake error printed in the
terminal, fix the missing prerequisite, then rerun the same command. Incremental
builds retain completed work.

See [live controls and local API](LIVE.md) for workload profiles, telemetry and
control semantics, or the [README](../README.md) for batch workloads,
transformer execution, TLM and architecture experiments.
