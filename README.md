# Analysing effect of gaps on structure functions

This set of codes performs an analysis of the effect of data gaps on the structure function as calculated from time series of the solar wind. It uses an HPC for running the full analysis, however, instructions for running it locally on a subset of the data are also provided below.

Further information about the data and methods described below are given in the [accompanying manuscript](https://arxiv.org/abs/2412.10053).

Please feel free to download the dataset and perform your own analysis or adapt it for your needs. For any comments or questions, either use the GitHub functionality or email daniel.wrench@vuw.ac.nz


## Data description

Data is downloaded from NASA's Space Physics Data Facility, as described in the instructions below. The 3 datasets are as follows:

### TRAINING
PSP magnetic field data (magnetometer instrument), 2019-2023 (4 years) = 10,731 standardised intervals.

### TESTING
Wind magnetic field data (magnetometer instrument)
**In submitted paper:** 2016-03-01 to 2016-12-17 (3.5 months) = 165 standardised intervals.

**Now have 7.5 months worth (225 processed files)**, 200 of which successfully corrected (resource reqs not quite enough for other 25 - won't worry about for now) = 441 standardised intervals.

### APPLICATION
Voyager 1 magnetic field data (magnetometer instrument). Two intervals:
- 118au
- 154au

## Setting up environment

> **`bolded code`** indicates commands you should type directly into the terminal.

(It should be relatively easy to adjust to use CDF files from other spacecraft as well, mainly via editing the `src/params.py` parameter file.)

In order to create the full, multi-year dataset, an HPC cluster is required. However, for the purposes of testing, minor adjustments can be made to the pipeline so that it can be run locally on your machine with a small subset of the data: note some local/HPC differences in the instructions below. This local version has been run on a **Windows OS**. Both the HPC and local versions use **Python 3.10.4**.

**Google Colab** is a highly recommended way to run the code for beginners on a Windows computer. 
You will need to prefix the commands below with `!`, use `%cd` to move into the project folder, and can safely ignore step 2.

1. **Clone the repository to your local machine**

    - Using a terminal: `git clone https://github.com/daniel-wrench/sf_gap_analysis`

    - Using GUI in VS Code: [see here for instructions](https://learn.microsoft.com/en-us/azure/developer/javascript/how-to/with-visual-studio-code/clone-github-repository?tabs=create-repo-command-palette%2Cinitialize-repo-activity-bar%2Ccreate-branch-command-palette%2Ccommit-changes-command-palette%2Cpush-command-palette#clone-repository)

2. **Create a virtual environment** 

    *Local:* 
    1. **`python -m venv venv`**
    2. **`venv/Scripts/activate`**

    *HPC:*
    1. **`module load Python/3.10.4`**
    2. **`python -m venv venv`**
    3. **`source venv/bin/activate`**

2. **Install the required packages**

    **`pip install -r requirements.txt`**

3. **Download the raw CDF files using a set of recursive `wget` commands**

    *Local:* **`bash 0_download_files.sh`**

    *Run on tmux on terminal to do other stuff while it downloads. Alternatively the `sunpy` package can be used for selecting specific time ranges more easily, but here we simply want to download all available data.*

    **PSP:** Cleaning all the available data (5 and a bit years), we go from 6000 files (130GB) -> 4380 files (3.5 years)

    **Wind:** Performing the same filtering on all the Wind data from 2016, we 5 months of data


4. **Delete files that are too small**

   **`bash delete_small_files.sh`**

    Not all files contain the full 6 hours expected: this script deletes any files smaller than 1MB.

## Processing data

> **`bolded code`** indicates commands you should type directly into the terminal.

1. **Process the data, file by file**

    *Demo?*

    *Description, including key decisions:*

    1. Extract vector magnetic fields from CDF files
    2. If more than 20% is missing from the file, skip, and add to a list of bad files `failed_files.txt` and remove it from the data directory. Later, we also add to this list and remove any files where the dataset time is less than the number of lags we want to compute up to, and any files where no complete standardised intervals can be extracted.
    3. Set the parameters (approx corr time, cadence, and n_lags to compute up to) pertaining to that dataset (Wind or PSP)
    4. Standardise # corr times (10) contained within each 10,000 pt. resampled interval based on (integral) corr time of the full dataset
    5. For std ints with >1% missing, delete. Otherwise, linearly interpolate gaps.
    6. Save some preprocessing plots showing the standardisation process for each file.
    7. Calculate the 2nd order SF up to 20% of the interval length (2000 lags), as well as the slope over a given range (50-500 lags)
    8. Calculate the corresponding ACF and corr scale (1/e)
    9. Gap the input intervals both uniformly and in chunks multiple times. 
    10. Calculate the SF and slope from the gapped interval
    11. Linearly interpolate the gapped interval and calculate the SF and slope
    12. Save outputs to a dictionary pickle file, with file name the same as the input file that the stats we calculated from.

    *Running locally (with a subset of the data):*

    1. In `src/params.py`, adjust `data_prefix_path`, depending on where you are storing the data (if local, likely in code dir, so set to `""`), and `times_to_gap` as well

    1. In `1_compute_sfs.sh`, change `start_index` to 0 
    
    2. **`bash 1_compute_sfs.sh`**

    *Running on an HPC (required for full dataset):* 
    
    1. In `src/params.py`, adjust `data_path_prefix`, depending on where you are storing the data, and likely `times_to_gap` as well

    2. In `1_compute_sfs.py`, comment out matplotlib params section as indicated after library imports


    2. Set job resource requests in `1_compute_sfs.sh`:
        - CORES: As many as possible
        - TIME: 20min/file (e.g. put on for 6 hours if running on 10 files/core. Some files will be done much more quickly if fewers std ints can be extracted, but that's OK)
        - MEM: 1GB/core, regardless of # files
        - *If some jobs do time out, meaning some files remain unprocessed, we can limit the `data/raw` directory to those unprocessed files by moving them with `python move_matching_files.py`. Maybe in future make this part of the slurm/python job*

    2. **`sbatch 1_compute_sfs.sh`**
    

2. **Perform train-test split for PSP data.** *Be very careful about whether you want to delete existing files from the train-test folders. Currently these lines are commented out.*

    (If on HPC, make sure you have `module load`ed Python first)

    **`python 2_train_test_split.py`**


3. **Assign the errors to each bin for each file**

    *Output*:

    `data/processed/psp/train/errors/[file_name]_pe_[dim]_[bins]_[naive/lint].pkl`

    *Local:*

    1. In `3_bin_errors.sh`, change `start_index` to 0 
    
    2. **`bash 3_bin_errors.sh`**

    *HPC:* 

    1. Set job resource requests in `3_bin_errors.sh`:
        - CORES: As many as possible
        - TIME: 15s/file/core
        - MEM: 500MB/core, regardless of # files

    2. In `3_bin_errors.py`, adjust `data_prefix_path`, depending on where you are storing the data

    3. **`sbatch 3_bin_errors.sh`**

4. 

- 4a. **Merge the binned errors and calculate the correction factor**  

    *Output:* 
    - `data/corrections/[path]/correction_lookup_[dim]_[bins]_lint.pkl` (for correction and plotting)
    - `data/corrections/[path]/correction_lookup_[dim]_[bins]_naive.pkl` (for plotting)
    - Some interim plots for inspection

    *Local:* **`bash 4a_finalise_correction.sh`**

    *HPC:* 

    1. Set job resource requests in `4a_finalise_correction.sh`
        - CORES: 1 (serial job)
        - MEM: see below
        - TIME: see below
        - 100 files "" = 350MB, 120s
        - 200 files "" = 500MB, 150s
        - 400 files "" = 820MB, 210s
        - 1000 () files "" = 1.7G, 7min
        - **4200 (all) files "" = 7G, 32min**

    2. **`sbatch 4a_finalise_correction.sh`**

- 4a. i. **Smooth heatmaps** to get less jumpy corrections

    **`python 4a_i_smooth_correction.py`**

- 4b. **Calculate stats of the training set**

    *Output:* 
    
    - If `include_sfs=False`, for the whole set, get average slope and file corr time `training_stats.txt`
    - If `include_sfs=True`, output lag-specific errors: `data/processed/psp_train_sfs_gapped.pkl`. These are later plotted in step 7b. **NB:** need to limit the number of files with `n_files` param in python script, beneath `include_sfs` param. We can do *at least* 20 files.

    *Local:*

    **`bash 4b_compute_training_stats.sh`**

    *HPC:*

    1. Set job resource requests in `4b_compute_training_stats.sh`

    2. **`bash 4b_compute_training_stats.sh`**

- 4c. **Plot the heatmaps and error trendlines for the training set** (full dataset and subset respectively)

    *Output:* `plots/results/train_psp_error.png`

    1. Download the outputs from step 4a.

    2. **`python 4c_plot_training_results.py`**


5. **Perform the correction on the test set, file by file**

    And also calculates the slopes, correlation scales, taylor scales from all the SF estimates. Before this, they had only been calculated for the true SF back in `1_compute_sfs.py` *(which is not strictly necessary any more, other than making preprocessing plots for interest)*.

    - NOTE ALSO DIFFERENT VERSIONS OF SF_FUNCS.LOAD_AND_CONCATENATE?
    - Cannot correct PSP right now, only Wind. This is due to file output paths: PSP has `psp/train,test`, Wind does not.

    *Local:* 
    
    1. If you are after minimal output from the full dataset, set `full_output = False`. (These files, used by the next script, go into `data/processed`.) If you are after full output (from just a few intervals) for later plotting in case studies, set `full_output = True`. (These files go into `data/corrections`.)
    
    **`for i in $(seq 0 2); do python 5_correct_test_sfs.py $i; done`**

    *HPC:* 
    1. Set job resource requests:
        - 2GB AND 3MIN/FILE

    2. **`sbatch 5_correct_test_sfs.sh`**

6. **Compute the statistical results for all (corrected) test set files**

    *Local*: **`bash 6_compute_test_stats.sh`**

    *HPC*:

    1. Set job resource requests:
        - 30s and 100MB for the full 111 Wind files, containing 125 intervals

    2. **`sbatch 6_compute_test_stats.sh`**

    *Output*: 
    - `test_corrected_{spacecraft}_{bins}_bins.pkl`, not including `ints`, `ints_gapped`, `sfs`, or `sfs_gapped_corrected`
    - `test_corrected_{spacecraft}_{bins}_bins_corrs.csv`
    - `test_corrected_{spacecraft}_{bins}_bins_stats.csv`

## Plotting results

7.  **Plot the test set results** 

    1. If on an HPC, download the following files first:
    - all outputs in `sf_gap_analysis/data/corrections`, including heatmaps
    - first few corrected files in `/nesi/nobackup/vuw04187/data/processed/wind` to plot as case studies
         
    2. **`python 7a_plot_test_results.py {spacecraft} {n_bins}`** (scatterplots, boxplots)

    3. **`python 7b_plot_test_case_studies.py  {spacecraft} {n_bins}`**

    4. **`python plot_scalar_dists.py slope > scalar_dists_tests.txt`**

Final publication-ready plots are then moved to `figs/`

## Notes/next steps

- Create correction factor file
- Lines 367-387 in `1_compute_sfs.py` aren't strictly necessary, as we compute all these derived stats now in step 5. However, those stats are used in the preprocessing plots that are output by this script.
- Add line about data size, e.g. *The HPC version of the code currently ingests 300GB across 10,000 CDF files (data from 1995-2022) and produces an 18MB CSV file.*
- Include typical duration (range) of standardised intervals for each spacecraft 
- Normal**iz**ation for Voyager plots (better pipeline in that script)
- Better case-study examples - using quicker way to view many of them?
- Clarify effect of standardisation in limitations section, as Mark mentioned
- Highlight emphasise on good overall shape, rather than inertial range slope, based on results?
- Previous slope range (1-10\% of corr time) did give results that matched theoretical values well, e.g. median of 0.67 from 175 PSP ints, 0.72 for 40 Wind ints
- Calculate sf_2_pe in 1_compute_sfs? Currently not to have somewhat simpler calculation once corrected, but also leading to some duplication of code, especially if we want the error trend line plots.
- Add handling, e.g. in sf func, for extreme cases where SF will be missing values for certain lags due to high % missing (not a high priority for now because only going up to lag 2000, e.g. still 30 dx values for 99.6% missing)
- Having logarithmically spaced lag bins would make the correction factor much cleaner to work with: one-to-one bins
- For now likely to do stick with simple job arrays and single jobs on HPC, with importing and exporting of intermediate steps, but perhaps better to do single MPI script with broadcasting and reducing.
- *CORRECTION CASE STUDIES plot*:
    - Confirm conf intervals
    - Add nice little box for slope errors with annotations, or else some way to comment on slopes
    - Smooth the results, check it works with these two corrections. Update error accordingly

