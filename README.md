# DOTs-SOCP

**DOTs-SOCP** contains the implementation for solving the dynamical optimal transport (DOT) problem on discrete surface. This repository facilitates research on the algorithm for solving the Second-Order Cone Programming (SOCP) reformulation of DOT problem on triangulated surfaces. It is designed for easy replication of all numerical experiments in our paper.

### Contact

E-mail: chl@hnu.edu.cn
Home page: https://grzy.hnu.edu.cn/site/index/chenliang3

### Copyright

The version of DOTs-SOCP is distributed under the GNU AFFERO GENERAL PUBLIC LICENSE, version 3. A copy can be found in the `LICENSE` file.

## Repository Structure

```
.
├── dot_surface_socp/  # Core implementation of DOTs-SOCP
├── replication/       # Scripts for replication
├── demo.py            # Demo script for basic usage
└── Makefile           # One-click command of replication
```

The repository is organized into several main components:

*   `dot_surface_socp/`: Contains the core logic, algorithm, and utilities for the DOTs-SOCP method.
*   `replication/`: Includes implementations of scripts for experiment execution and result analysis.
*   `demo.py`: A script to demonstrate the basic usage.
*   `Makefile`: Provides convenient commands for running all experiments with one-click.

## Installation

### Setup python environment

Using `uv` (Recommended).
1. Install `uv` (if not already installed) using:
    ```bash
    pip install uv
    ```
2. Synchronize Python virtual environment using:
    - **Windows:**
        ```bash
        uv sync --extra windows
        ```
    - **Linux:**
        ```bash
        uv sync --extra linux
        ```

Alternatively, using `pip`.
1. Create a Python 3.12 virtual environment:
    ```bash
    python -m venv .venv
    ```
2. Activate the environment:
    - **Windows:**
        ```bash
        .venv/Scripts/activate
        ```
    - **Linux:**
        ```bash
        source .venv/bin/activate
        ```
3. Install dependencies using pip:
    - **Windows:**
        ```bash
        pip install -r requirements_windows.txt
        ```
    - **Linux:**
        ```bash
        pip install -r requirements_linux.txt
        ```

### Graphics Dependencies (Linux only)

If running on a Linux system without graphics dependencies, install OpenGL and Mesa:
```bash
sudo apt-get update
sudo apt-get install -y libosmesa6-dev libgl1-mesa-dev
```

### Make (Windows only)

To use the commands for one-click experiment replication, install `make` by running the following command in `Powershell`:
```powershell
winget install GnuWin32.Make; [Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";C:\Program Files (x86)\GnuWin32\bin", "User"); $env:PATH += ";C:\Program Files (x86)\GnuWin32\bin"
```

## Running Experiments

### One-Click Replication

For easy replication of all numerical experiments, we provide convenient one-click command:

```bash
make all
```

This command will automatically run all experiments and save the results in the `output/` directory. For more details on the make commands, run `make help`.

### Individual Experiment Commands

One can run a specific experiment using `demo.py` with the following optional arguments:
- **Problem Settings**:
    1. `--example`: Specifies the example to run.
    2. `--congestion`: Sets the congestion factor. (Default: $0.0$)
- **Algorithm Parameters**:
    1. `--ntime`: The number of points in the discrete time grid (Default: $31$)
    2. `--nit`: The maximum number of iterations (Default: $10^3$)
    3. `--tol`: The convergence tolerance for iteration. (Default: $10^{-3}$)
- **Output Options**:
    1. `--show`: Displays the animation in an interactive window. (Default: `True`)
    3. `--save`: Saves the animation (as a series of `.png` files and a `.mp4` video). (Default: `False`)

For example, run experiment of `knots_5` example using:

```shell
python demo.py --example=knots_5 --show
```

For more details on available arguments, run `python demo.py --help`.
