# LowPrecisionApproxGP
Github repository for Low Precision Approximate Gaussian Process Inference - a group project for CDS's Bayesian Machine Learning class - created by Giulio Duregon, Jonah Poczobutt, and Paul Kottering.

## Workflow:
```bash
# Set up enviroment variables, pull in data
> source setup.sh
> bash scripts/getData.sh
# Run Experiments ad hoc
> python model_runner.py -l True -d <dataset> -dt <data_type> -bk <base_kernel> ...
# Parse the outputs ad hoc
> python scripts/parse_logs.py
# Or setup run_experiments.sh as you like
> bash scripts/run_experiments.sh
```

## TODO: 
Setup:
- - Team Decision: Encorporate size of the working set for experimentation
$$J \subset \{n-m\}$$
- Make normalizing function?
- Test loss -- DONE
- Set Random seed for reproducibility -- DONE
- Set all tensors to device
- Run on GPU

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
| `ip` | `max_inducing_points` |  Maximum number of inducing points to be added. |
| `dt` | `datatype` | Torch.dtype to be used in computation. |
| `s` | `save_model` |  Boolean to save model. |
| `sfp` | `save_model_file_path` |  Output destination for saving model post training. |
| `l` | `logging` |  Boolean for enabling logging. |
|`lop` | `logging_output_path` |  Model Training logging output destination file path. |
|`m` | `use_max` | Instructs the model to use O(n) search to find max MLL inducing point to be selected|
|`j` | `j` | #TODO: Size of subset of inducing point candidates|
| `mj` | `max_js` | Maximum number of subsets considered when searching for an inducing point to increase MLL|
|||