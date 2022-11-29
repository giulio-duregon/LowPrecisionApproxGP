# LowPrecisionApproxGP
Github repository for Low Precision Approximate Gaussian Process Inference - a group project for CDS's Bayesian Machine Learning class - created by Giulio Duregon, Jonah Poczobutt, and Paul Kottering.

## Workflow:
```bash
> source setup.sh
> bash scripts/getData.sh
```

## TODO: 
Setup:
- Giulio: Finish setting up **Inducing Point Kernel/Added Loss/Strategy**
- Paul: Set up alternative greedy select points function that finds the best out of all candidates -> Copy the `/LowPrecisionApproxGP/util/GreedySelector.py` and make it find the best MLL rather than just one that increases the likelihood
- - Team Decision: Encorporate size of the working set: 
$$J \subset \{n-m\}$$
- Giulio: Set up testing for variable precision torch dtypes and assert that the return values conform to these dtypes
- Jonah + Giulio: Finish setting up experiment_runner

## Maybe if we use HPC
- Figure out which overlays (or how to make them) are necessary for running a container
- Get script to download overlay and launch singularity container
- Update `dataIntoRam.sh` script to point to saved file location on cluster