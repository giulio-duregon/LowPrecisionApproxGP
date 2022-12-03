# LowPrecisionApproxGP
Github repository for Low Precision Approximate Gaussian Process Inference - a group project for CDS's Bayesian Machine Learning class - created by Giulio Duregon, Jonah Poczobutt, and Paul Kottering.

## Workflow:
```bash
# Set up enviroment variables, pull in data
> source setup.sh
> bash scripts/getData.sh
# Run Experiments
> python LowPrecisionApproxGP/model_runner.py -l True ## TODO: Finish this with all the args

```

## TODO: 
Setup:
- Giulio: Finish setting up **Inducing Point Kernel/Added Loss/Strategy** -- DONE
- Paul: Set up alternative greedy select points function that finds the best out of all candidates -> Copy the `/LowPrecisionApproxGP/util/GreedySelector.py` and make it find the best MLL rather than just one that increases the likelihood
- - Team Decision: Encorporate size of the working set: 
$$J \subset \{n-m\}$$
- Giulio: Set up testing for variable precision torch dtypes and assert that the return values conform to these dtypes -- DONE
- Jonah + Giulio: Finish setting up experiment_runner
- - Make sure can't select float16 when on CPU -- DONE
- - Better logging of results, what scheme makes sense -- DONE
- Dataset factory loader --DONE
- Make normalizing function?
- Make Log results parser

## Maybe if we use HPC
- Figure out which overlays (or how to make them) are necessary for running a container
- Get script to download overlay and launch singularity container
- Update `dataIntoRam.sh` script to point to saved file location on cluster


## Variables / Arguments Explanation
| Short Name | Long Name | Description |
| :------------ | :------------ |  :-----------: |
| `d` | `dataset` | Specifies dataset to use. |
| `bk` | `base_kernel_type` | Selects the base kernel to be used in model. |
| `it` | `training_iter` |  Maximum number of training/optimization iterations. |
| `dt` | `datatype` | Torch.dtype to be used in computation. |
| `s` | `save_model` |  Boolean to save model. |
| `sfp` | `save_model_file_path` |  Output destination for saving model post training. |
| `l` | `logging` |  Boolean for enabling logging. |
|`lop` | `logging_output_path` |  Model Training logging output destination file path. |