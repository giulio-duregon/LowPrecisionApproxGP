# LowPrecisionApproxGP
Github repository for Low Precision Approximate Gaussian Process Inference - a group project for CDS's Bayesian Machine Learning class - created by Giulio Duregon, Jonah Poczobutt, and Paul Kottering.

## Workflow:
```bash
> source setup.sh
> bash scripts/getData.sh
```

## TODO: 
Setup:
- Read up on when to use different MLLs 
- Create Half **Inducing Point Kernel** SGPR Docs / InducingPointKernelAddedLossTerm
- Figure out which overlays (or how to make them) are necessary for running a container
- Get script to download overlay and launch singularity container
- Update `dataIntoRam.sh` script to point to saved file location on cluster

Modeling: 
- Figure out how to change precision as necessary for experiments