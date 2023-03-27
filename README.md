# SwimFunction

This repository preprocesses videos, calculates recovery metrics, and predicts recovery outcomes as described in

Jensen et al., Functional trajectories during innate spinal cord repair. bioRxiv doi: https://doi.org/10.1101/2023.01.31.526502

## Installation

Make sure you have python installed on your computer and the command `conda` is accessible either with [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

1) Clone this repository to your computer.

2) Create the virtual environment:

        conda env create --name swimfunction --file=swimfunction-env.yaml

3) Activate the virtual environment:

        conda activate swimfunction

4) Install the cloned repository via pip:

        pip install /path/to/SwimFunction

5) One plotting function (not required for metric calculation) requires both R and a forked version of the [sigclust2](https://github.com/pkimes/sigclust2) repository located [here](https://github.com/discolemur/sigclust2). You can install it in R by running this command in the R interpreter:

        devtools::install_github("discolemur/sigclust2")

Each time you use this package, be sure to activate the virtual environment: `conda activate swimfunction`

## Setting up your experiment

Description will go here.

## Required inputs

Description will go here.

### Integrating precalculated metrics

CRITICAL: All glial bridging and axon regrowth scores are assumed to be percents between 0 and 1 (where 1 is 100%). If the score files you provide include scores between 1 and 100 (where 100 is 100%), some plotting functions may misbehave.

### Expected outputs

Description will go here.
