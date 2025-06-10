# DOTs-SOCP -- a Python software for dynamic optimal transport problem on surfaces

### Liang Chen, Youyicun Lin, and Yuxuan Zhou

This open-source software package provides an efficient numerical optimization approach for solving dynamic optimal transport (DOT) problems on general smooth surfaces. It computes the quadratic Wasserstein distance and associated transportation paths by solving a linear second-order cone programming (SOCP) reformulation. The implementation utilizes an inexact semi-proximal augmented Lagrangian method.

This repository enables the straightforward replication of numerical results for our proposed method, as presented in the paper. It offers a robust and highly efficient solution for DOT problems on surfaces. Additionally, this codebase aims to serve as a valuable foundation for future research in optimal transport and related numerical methods.

### Citation

* **Liang Chen, Youyicun Lin, and YuXuan Zhou**, An efficient augmented Lagrangian method for dynamic optimal transport on surfaces based on second-order cone programming, preprint, 2025.

* **Important note:**

  * The software is still under development, so it will invariably be buggy. We would appreciate your feedback and bug reports.

  * This is research software. It is not currently intended or designed to be general-purpose software.
 
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

### Setup Python environment

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

To use the commands for one-click experiment replication, install `make` by running the following command in `PowerShell`:
```powershell
winget install GnuWin32.Make; [Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";C:\Program Files (x86)\GnuWin32\bin", "User"); $env:PATH += ";C:\Program Files (x86)\GnuWin32\bin"
```

## Running Experiments

### One-Click Replication

For easy replication of all numerical experiments, we provide a convenient one-click command:

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

For example, run the experiment of `knots_5` example using:

```shell
python demo.py --example=knots_5 --show
```

For more details on available arguments, run `python demo.py --help`.
