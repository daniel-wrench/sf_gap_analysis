# Gaps on structure functions

This set of codes performs an analysis of the effect of data gaps on the structure function as calculated from time series of the solar wind. It uses an HPC for running the full analysis; however, instructions for running a smaller version locally are also provided below.

Further information about the data and methods described below are given in the accompanying manuscript, which will be linked here upon publication.

## Data

TRAINING: PSP magnetic field data.

TESTING: Wind
- 111 raw files (1 day each) = 3.5 months obtained from 2016-03-01 to 2016-12-17
- Returns 660 standardised intervals (some days have more than one), gapped 25 times each to give **16,500** testing intervals.

## How to run this code

(It should be relatively easy to adjust to use CDF files from other spacecraft as well, mainly via editing the `src/params.py` parameter file.)

In order to create the full, multi-year dataset, an HPC cluster is required. However, for the purposes of testing, minor adjustments can be made to the pipeline so that it can be run locally on your machine with a small subset of the data: note some local/HPC differences in the instructions below. This local version has been run on a **Windows OS**. Both the HPC and local versions use **Python 3.10.4**.

**Google Colab** is a highly recommended way to run the code for beginners on a Windows computer. 
You will need to prefix the commands below with `!`, use `%cd` to move into the project folder, and can safely ignore step 2.

1. **Clone the repository to your local machine**

    - Using a terminal: `git clone https://github.com/daniel-wrench/reynolds_scales_project`

    - Using VS Code: [see here for instructions](https://learn.microsoft.com/en-us/azure/developer/javascript/how-to/with-visual-studio-code/clone-github-repository?tabs=create-repo-command-palette%2Cinitialize-repo-activity-bar%2Ccreate-branch-command-palette%2Ccommit-changes-command-palette%2Cpush-command-palette#clone-repository)

2. **Create a virtual environment** 

    Local: 
    - `python -m venv venv`
    - `venv/Scripts/activate`

    HPC:
    - `module load Python/3.10.4`
    - `python -m venv venv`
    - `source venv/bin/activate`

2. **Install the required packages**

    `pip install -r requirements.txt`

3. **Download the raw CDF files using a set of recursive `wget` commands**

    Local: `bash 0_download_files.sh`

    (Run on tmux on terminal to do other stuff while it downloads)

    About 6000 files for the full 5 and a bit years of 6-hourly data from PSP = 130GB of disk space. After filtering done in the next two steps, **we end up with 4380 files (3.5 years worth)** = amount of clean data available from existing 5 and bit years of data, given current conditions. Performing the same filtering on all the Wind data from 2016, we 5 months of data


4. **Delete files that are too small**

    `bash delete_small_files.sh`

    A common error is that the file is unexpectedly small: i.e., it is not the full 6 hours, or at least does not contain 10,000 points at the initial resampled cadence. **We should run a simple check for this prior, deleting any files smaller than 1MB (and then seeing whether too many/not enough are caught by this, depending on any remaining error messages)**.

4. **Process the data, file by file**

    In `src/params.py`, adjust `data_prefix_path`, depending on where you are storing the data (if local, likely in code dir, so set to `""`), and likely `times_to_gap` as well

    Local:

    In `1_compute_sfs.sh`, change `start_index` to 0 
    
    `bash 1_compute_sfs.sh`

    HPC: 
    
    Adjust `data_prefix_path`, depending on where you are storing the data

    `sbatch 1_compute_sfs.sh`
    
    HPC requirements:
    - Average of 20-40min/file: e.g. put on for 6 hours if running on 10 files/core
    - Only ever need 2GB per core
    - If some jobs do time out, meaning some files remain unprocessed, we can limit the `data/raw` directory to those unprocessed files by moving them with `python move_matching_files.py`. *Maybe in future make this part of the slurm/python job*

    - Processed PSP and Wind files are between 32 (~400MB used in step 1) and 156MB (~300MB) each. For PSP, there are an average of 4 files/file

    *Might have demo in the old time series repo*

3. Perform train-test split for PSP data. **Be very careful about whether you want to delete existing files from the train-test folders. Currently these lines are commented out.**

    (make sure you have `module load`ed Python if on an HPC first)

    `python 2_train_test_split.py`


4. **Assign the errors to each bin for each file**

    Local:

    In `3_bin_errors.sh`, change `start_index` to 0 
    
    `bash 3_bin_errors.sh`

    HPC: 
    
    In `3_bin_errors.py`, adjust `data_prefix_path`, depending on where you are storing the data

    `sbatch 3_bin_errors.sh`

    - LATEST: 20 files/core, {2d, 3d} {15,20,25 bins} = 300MB, 3.5min
    - 5o files/core "" = CONSTANT 500MB, no matter how many files, 15s/file
    - Basically 15min to do the whole lot across 60 cores (73 files/core)

4. **Merge the binned errors and calculate the correction factor**

    Local: `bash 4a_finalise_correction.sh`

    HPC: `sbatch 4a_finalise_correction.sh`

    LATEST: 10 files (SERIAL JOB), {2d, 3d} {15,20,25 bins} = 200MB, 90s

    100 files "" = 350MB, 120s
    200 files "" = 500MB, 150s
    400 files "" = 820MB, 210s
    1000 () files "" = 1.7G, 7min


    - **4200 (all) files "" = 7G, 32min**

5. **Calculate the stats (average slope and corr time, error trend lines) for the training set**

    `bash 4b_compute_training_stats.sh`

    **NB**: Limit the number of files, as we will not be able to plot the error trendlines locally in step 7b on the full dataset. We can do *at least* 20 files.

5. **Perform the correction on the test set, file by file**

    *111 Wind files to correct: outputs are 12-22KB each = 3MB files**

    THERE IS A VERSION FOR SAVING THE CORRECTED SFS, FOR COMPUTING OVERALL TEST RESULTS, AND A VERSION WITHOUT, FOR THE CASE STUDY PLOTS, TO BE RUN LOCALLY. NOTE ALSO DIFFERENT VERSIONS OF SF_FUNCS.LOAD_AND_CONCATENATE

    Local: `for i in $(seq 0 1); do python 5_correct_test_sfs.py $spacecraft $i $n_bins; done`

    HPC: `sbatch 5_correct_test_sfs.sh`

    1GB AND 3MIN/FILE

6. **Compute the statistical results for all (corrected) test set files**

    `bash/sbatch 6_compute_test_stats.sh`

    Reqs: 

    (Now using simplified outputs)
    - 30s and 100MB for the full 111 Wind files, containing 125 intervals

    **Output: test_corrected_{spacecraft}_{bins}_bins.pkl** *not including ints, ints_gapped, sfs, or sfs_gapped_corrected*

7. IF ON HPC, DOWNLOAD THE FOLLOWING FILES:

    - all outputs in `sf_gap_analysis/data/processed`
    - first few corrected files in `/nesi/nobackup/vuw04187/data/processed/wind`

7.  **Plot the test set results**
     (If on an HPC, download the above output at this step, as well as the heatmaps  and the **FIRST**  2-3 individual corrected pickle files for plotting case studies from) 
    `python 7a_plot_test_overall.py {spacecraft} {n_bins}`

    `python 7b_plot_test_case_studies.py  {spacecraft} {n_bins}`


### Notes/next steps
- Better case-study examples.
- Normal**iz**ation for Voyager plots (better pipeline in that script)
- Clarify effect of standardisation in limitations section, as Mark mentioned
- Lockwood (2019) work is useful complement, showing that gaps cannot simply be ignored for our other time-domain stat, the ACF
- Previous slope range (1-10\% of corr length) did give results that matched theoretical values well, e.g. median of 0.67 from 175 PSP ints, 0.72 for 40 Wind ints
- Calculate sf_2_pe in 1_compute_sfs? Currently not to have somewhat simpler calculation once corrected, but also leading to some duplication of code, especially if we want the error trend line plots.
- Add handling, e.g. in sf func, for extreme cases where SF will be missing values for certain lags due to high % missing (not a high priority for now because only going up to lag 2000, e.g. still 30 dx values for 99.6% missing)
- Would be nice to get total # intervals for each set returned by step 1
- Wind data reads very slowly, compared with PSP. It is using a pipeline function that I think Kevin made, made up of many smaller functions.
The bottleneck is the "format epochs" function. I've starting trying to do this in the same was as PSP, but it was struggling to do the timedelta addition
- Having logarithmically spaced lag bins would make the correction factor much cleaner to work with: one-to-one bins
- For now likely to do stick with simple job arrays and single jobs on HPC, with importing and exporting of intermediate steps, but perhaps better to do single MPI script with broadcasting and reducing.
- *CORRECTION CASE STUDIES plot*:
    - Confirm conf intervals
    - Add nice little box for slope errors with annotations, or else some way to comment on slopes
    - Smooth the results, check it works with these two corrections. Update error accordingly
- Add line about data size, e.g. *The HPC version of the code currently ingests 300GB across 10,000 CDF files (data from 1995-2022) and produces an 18MB CSV file.*
