# IC inequality

This repository contains the source code, aggregate empirical data and simulation results for our manuscript **Older adults experience greater information narrowing on a short-video platform**.

We study how exploration and exploitation interact with personalized recommendation to shape age-related inequality in information consumption. The code is implemented in Python.

## System requirements

Python 3.11. The required packages and their versions are listed in [requirements.txt](requirements.txt).

## Installation

Clone the repository and install the dependencies:

```sh
git clone https://github.com/Piaojh97/IC_inequality.git
cd IC_inequality
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Demo and reproduction

### 1. Reproduce figures and tables

Run the following command to draw the numerical panels of Figures 1–4, prepare Supplementary Figures S1–S17, and export Supplementary Tables S1–S24 from the supplied data:

```sh
python code/reproduce.py
```

Figures are saved as PDF and PNG files in `outputs/figures/`, and tables as CSV and Markdown files in `outputs/tables/`. The original illustrations for Figure 2f,g and Supplementary Figure S10 are supplied with the figure files.

To reproduce a particular figure or the supplementary tables:

```sh
python code/reproduce.py --main 1
python code/reproduce.py --supplementary 12
python code/reproduce.py --tables
```

| Main figure | Content | Reproduction command |
|---|---|---|
| Figure 1 | Age-related inequality in information-cocoon risk | `python code/reproduce.py --main 1` |
| Figure 2 | Exploration and exploitation across age groups | `python code/reproduce.py --main 2` |
| Figure 3 | Information-dynamics model and empirical agreement | `python code/reproduce.py --main 3` |
| Figure 4 | Interventions to reduce the age gap | `python code/reproduce.py --main 4` |

The [figure and table guide](docs/FIGURE_GUIDE.md) lists the code and data for every main panel, supplementary figure and supplementary table. The [reproduction notes](docs/REPRODUCTION.md) describe the plotting conventions and model settings.

### 2. Run a model demo

Run a small simulation with 12 users, 200 interaction steps and 200 candidate items:

```sh
python code/model/run.py --mode demo --users 12 --steps 200 --items 200
```

The output, `outputs/simulation/demo.csv`, reports the simulated group's risk, viewing diversity, persistence and depth of information cocoons. The run settings are saved alongside the results.

### 3. Run the simulation experiments

The following commands use the study's settings of 5,000 interaction steps, 10,000 candidate items and ten recommendations per step.

```sh
# Risk landscape across 100 exploration–exploitation combinations
python code/model/run.py --mode grid --users 1000 --betas 7

# Population simulation with 10,000 users
python code/model/run.py --mode population --users 10000

# Three interventions at eleven adjustment levels for the oldest group
python code/model/run.py --mode interventions --users 1596

# Parameter scan across nine recommendation strengths
python code/model/run.py --mode grid --users 1000 --betas 1,2,3,4,5,6,7,8,9
```

Simulation outputs are saved in `outputs/simulation/`. The full parameter scan evaluates 900 settings; its runtime depends on the number of users and interaction steps. Model parameters and the normalized intervention scale are described in the [reproduction notes](docs/REPRODUCTION.md).

## File description

| Folder | Contents |
|---|---|
| `code/` | Figure and table reproduction, data preparation and verification scripts |
| `code/model/` | Information-dynamics simulator and experiment runner |
| `data/empirical/` | Group statistics and distributions for the empirical figures |
| `data/statistical_results/` | Regression estimates, mediation results and supplementary statistics |
| `data/model/inputs/` | Topic-level inputs and age-group behavioural distributions |
| `data/model/outputs/` | Simulation risk landscapes, population summaries and intervention results |
| `data/tables/` | Supplementary Tables S1–S24 in CSV format |
| `reference/analysis/` | Original analysis notebooks and supplementary analysis scripts |
| `reference/figures/` and `reference/artwork/` | Manuscript figures and editable illustrations |
| `reference/table_sources/` | LaTeX table sources and the simulation algorithm |
| `docs/` | Figure guide, data dictionary, model settings and file provenance |

See the [data dictionary](docs/DATA_DICTIONARY.md) for variable definitions and units.

## Data availability

Empirical data are shared as aggregate statistics, including fitted regression and mediation estimates.

The simulator builds on our earlier work, *Human–AI Adaptive Dynamics Drive Emergence of Information Cocoons*. Attribution and licence information are provided in [LICENSES](LICENSES/README.md).
