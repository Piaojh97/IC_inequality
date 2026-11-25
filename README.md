# IC inequality

This repository contains the source code, aggregate empirical data and simulation results for our manuscript **Older adults experience greater information narrowing on a short-video platform**.

We study how exploration and exploitation interact with personalized recommendation to shape age-related inequality in information consumption. The code is implemented in Python.

## Installation

The code requires Python 3.11. Dependencies are listed in [requirements.txt](requirements.txt).

```sh
git clone https://github.com/Piaojh97/IC_inequality.git
cd IC_inequality
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Figures and tables

```sh
python code/reproduce.py
```

This draws the numerical panels of Figures 1–4, prepares Supplementary Figures S1–S17 and exports Supplementary Tables S1–S24. Figures are saved as PDF and PNG in `outputs/figures/`, and tables as CSV and Markdown in `outputs/tables/`.

To reproduce a single figure or only the tables:

```sh
python code/reproduce.py --main 1
python code/reproduce.py --supplementary 12
python code/reproduce.py --tables
```

The [figure and table guide](docs/FIGURE_GUIDE.md) maps every panel, supplementary figure and supplementary table to its code and data. Plotting conventions and model settings are in the [reproduction notes](docs/REPRODUCTION.md).

## Model simulations

Run a small demo:

```sh
python code/model/run.py --mode demo --users 12 --steps 200 --items 200
```

The output, `outputs/simulation/demo.csv`, reports the simulated group's risk, viewing diversity, persistence and depth of information cocoons.

Run the simulation experiments with the study's settings:

```sh
# Risk landscape across 100 exploration–exploitation combinations
python code/model/run.py --mode grid --users 1000 --betas 7

# Population simulation
python code/model/run.py --mode population --users 10000

# Three interventions at eleven adjustment levels for the oldest group
python code/model/run.py --mode interventions --users 1596

# Parameter scan across nine recommendation strengths
python code/model/run.py --mode grid --users 1000 --betas 1,2,3,4,5,6,7,8,9
```

Results are saved in `outputs/simulation/`. Model parameters are described in the [reproduction notes](docs/REPRODUCTION.md).

## File description

* `code/`: figure and table reproduction, data preparation and verification scripts
* `code/model/`: information-dynamics simulator and experiment runner
* `data/empirical/`: group statistics and distributions for the empirical figures
* `data/statistical_results/`: regression estimates, mediation results and supplementary statistics
* `data/model/inputs/`: topic-level inputs and age-group behavioural distributions
* `data/model/outputs/`: simulation risk landscapes, population summaries and intervention results
* `data/tables/`: Supplementary Tables S1–S24 in CSV format
* `reference/`: original analysis notebooks and scripts, manuscript figures, editable illustrations and LaTeX table sources
* `docs/`: figure guide, reproduction notes and [data dictionary](docs/DATA_DICTIONARY.md)

The simulator builds on our earlier work, *Human–AI Adaptive Dynamics Drive Emergence of Information Cocoons*. See [LICENSES](LICENSES/README.md) for attribution and licence information.
