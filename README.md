# DPI: Dynamic Preference Inference under Contextual Shifts

Official implementation of the ICLR 2026 paper
**[“Learning What Matters Now: Dynamic Preference Inference under Contextual Shifts”](https://arxiv.org/pdf/2603.22813)**.

DPI studies **dynamic preference inference under contextual shifts**, where an agent must infer time-varying preferences from interaction history and adapt its behavior when the underlying context changes.

## Contents

* [Installation](#installation)
* [Environments](#environments)
* [Project Structure](#project-structure)
* [Citation](#citation)
* [License](#license)

---

## Installation

### 1. Create a Conda environment

```bash
conda create -n dpi python=3.10 -y
conda activate dpi
```

### 2. Install DPI

```bash
python -m pip install -U pip
python -m pip install -e .
```

Core dependencies are specified in `pyproject.toml`, including:

* `torch`
* `gymnasium`
* `numpy`
* `scipy`
* `tqdm`
* `matplotlib`

### 3. MuJoCo / HalfCheetah support

To run [`Half-Cheetah-v0`](src/DPI/env/half_cheetah_env.py), install `mo-gymnasium`:

```bash
python -m pip install mo-gymnasium
```

> [!IMPORTANT]
> `mo-gymnasium` does not install dependencies for every environment family by default.
> To run MuJoCo environments, install the MuJoCo extras:
>
> ```bash
> python -m pip install "mo-gymnasium[mujoco]"
> ```
>
> Alternatively, install all optional environment dependencies with:
>
> ```bash
> python -m pip install "mo-gymnasium[all]"
> ```

---

## Environments

DPI currently includes three environments under [`src/DPI/env/`](src/DPI/env/):

### `Queue-v0`

A multi-objective queueing environment involving waiting and queue-management decisions.

* Reward dimension: (d=5)
* Includes deadline, energy, and morality-related objectives
* Supports event-triggered contextual shifts

### `Maze-v0`

A discrete grid-navigation environment with dynamically changing hazards.

* Reward dimension: (d=5)
* Includes multiple competing objectives
* Scripted events induce changes in the environment and preference context

### `Half-Cheetah-v0`

A multi-objective continuous-control environment based on MuJoCo and `mo-gymnasium`.

* Reward dimension: (d=3)
* Includes deadline and energy-related objectives
* Supports event-triggered contextual shifts

---

## Project Structure

```text
.
├── src/
│   └── DPI/
│       ├── agent/          # DPI agents
│       ├── core/           # Training utilities, PPO updates, GAE, etc.
│       ├── env/            # Queue, Maze, and HalfCheetah environments
│       ├── model/          # Preference encoder and actor-critic models
│       └── run/            # Command-line entry points
├── pyproject.toml
└── README.md
```

---

## Citation

If you find this project useful for your research, please cite:

```bibtex
@inproceedings{cao2026dpi,
  title     = {Learning What Matters Now: Dynamic Preference Inference under Contextual Shifts},
  author    = {Xianwei Cao and Dou Quan and Zhenliang Zhang and Shuang Wang},
  booktitle = {International Conference on Learning Representations},
  year      = {2026}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
