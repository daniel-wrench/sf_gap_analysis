# Gaps on structure functions
*Furthering numerically improving our estimates of solar wind statistics (Re was implementing on large dataset, this is actually developing a new way)*

## Latest results

- Processed 3.5 years worth of PSP data (5000 6-hour files) = amount of clean data available from existing 5 and bit years of data, given current conditions; and 5 months of Wind data (same for all of 2016)
- Computed heatmap for 50 days worth of PSP data (200 files)
    - Average slope of 0.55, average tc of 1100s = 3km
    - Average intervals/file = 3.5
- *Currently computing heatmaps for 8 months of PSP data (1000 files), testing scaling once more before full 2 years of training data*
- **Now, parallelise heatmap calculation. Correct 100 wind files and 100 PSP files to check the reqs for this. Then download, confirming local pipeline, and tidy up plots, while waiting for heatmap to run**.

- Trained on 20,000 intervals, 25x785 PSP intervals (=2.5 months, coming from first 300 processed files), 15,20,25 bins, outdated slope range *takes about 3.5 hours*


![alt text](train_psp_heatmap_10bins_3d_lint_lag.png) 
![alt text](train_psp_heatmap_10bins_3d_lint_missing.png) 
![alt text](train_psp_heatmap_10bins_3d_lint_power.png)


- Tested on 25x175 PSP intervals (=43 days)

![alt text](plots/temp/test_psp_scatterplots_25_bins.png)

- Tested on 25x40 Wind intervals (=20 days, coming from first 20 raw files, as above)

![alt text](plots/temp/test_wind_scatterplots_25_bins.png)

- Draft manuscript completed with these results; likely to be only minor updates with latest numbers and figs. Important to scale up analysis to populate heatmaps better, as well as using vector stats and better range (50-500 lags = 5-50% of $\lambda_C$)
- *NB: Previous slope range (1-10\% of corr length) did give results that matched theoretical values well, e.g. median of 0.67 from 175 PSP ints, 0.72 for 40 Wind ints*


## To-do
### Analysis

1. ~~Make very clear description of current state and next steps for Tulasi at meeting (and me), referring to my UN notebook~~
2. ~~Come up with more streamlined pipeline, mainly to speed up plotting at the end.~~
2. ~~Get correlations on each method in test set~~
2. ~~Delete outputs and test new pipeline with updated slope range and streamlined outputs~~
1. ~~Updating step 1~~
    - ~~Do consistent missing data check *before* calculating ACF~~
    - ~~Switch to vector integral scale~~
    - ~~Save name of bad files to list and move out of main data dir during initial processing~~
    - ~~Test gapping fn: can we actually get up to 95% missing?~~
    - ~~Switch to vector SF (but prob just plot radial component for case studies)~~
    - ~~Diagnose error with interpolation~~
    - ~~Get error messages in failed file~~
    - ~~Check 2019 psp bad wasn't missing 20% before resampling~~
2. ~~Test updated pipeline locally: should have more gaps now~~
2. ~~Switch to May-July for Wind test set, much cleaner than Jan~~
3. Scaling study on NESI, now with vector stats, updated slope range, S4, better parallel imports, stats and plots, streamlined outputs
    - ~~Process 300 files of Wind data~~
    - Use heatmap from subset to test remainder of pipeline, figure out time for steps 3+
3. Run on all PSP data we can (don't worry about calculating kurtosis as well for now)
4. **Choose #bins based on PSP data, report final results on Wind data**
3. Potentially investigate smoothing and error bars - do they look OK as is?
3. Plot lag x-axes in units of $\lambda_C$?
3. Think about how to study Frat's method, and verify Burger's results
4. **Kurtosis** analysis
11. Send completed draft manuscript to Tulasi. Print out, talk through with Marcus. Don't worry about Voyager just yet.
12. Implement Fraternale's sample size threshold for fitting slopes, and send to him
13. Read Ruzmaikin

### Manuscript and plots

*Chat with Tulasi when have final figures*

1. ~~Finish first draft of paper~~
2. For standardisation demo, use secondary y-axis; add to Overleaf
2. Change boxplot linetype
2. Note correlation between correlation scales from Reynolds
2. Depending on final results, prob remove slope APE from scatterplots, just have boxplots separately. Potentially make corrected taylor scale 
2. Check Google Doc, Notion for notes, comments
3. Improve variogram clouds plot: what are we saying that isn't already covered by case study plots. And whatever that is, make it clear with good examples (probably same length as analysis intervals) and make the style consistent 
2. Make consistent (Latex) font, and specify sizes to match specifications in Overleaf

### Notes
- Look at nn_seff files in more depth: likely want to run multiple files/core on step 3 due to low mem usage
- Processed PSP and Wind files are between 32 (~400MB used in step 1) and 156MB (~300MB) each
- ~~Add true SFs to case study plots. Will require re-jig of `ints` df~~
- ~~Get mean slopes from true SFs. Maybe move "inertial range" if consistent bias wrt 2/3~~
- Problem with first step is uneven times due to some files having no intervals, some having up to 4. Might be better to run on 3-5 files, spaced out (i.e. every 3rd file) in order to get more even times across jobs.
- Add handling, e.g. in sf func, for extreme cases where SF will be missing values for certain lags due to high % missing (not a high priority for now because only going up to lag 2000, e.g. still 30 dx values for 99.6% missing)
- Would be nice to get total # intervals for each set returned by step 1
- investigating bad ints and possible automatic removal during download or initial reading
- ~~consistency between times_to_gap across files~~
- Wind data reads very slowly, compared with PSP. It is using a pipeline function that I think Kevin made, made up of many smaller functions.
The bottleneck is the "format epochs" function. I've starting trying to do this in the same was as PSP, but it was struggling to do the timedelta addition
- Can add smoothing to correction step alter, **not on critical path for getting most of the scripts on NESI**
- Having logarithmically spaced lag bins would make the correction factor much cleaner to work with: one-to-one bins
- For now likely to do stick with simple job arrays and single jobs on HPC, with importing and exporting of intermediate steps, but perhaps better to do single MPI script with broadcasting and reducing.
- Might using actual pandas indexes be easier/more efficient?
- Add sf slope to Wind dataset

## How to run this code

(It should be relatively easy to adjust to use CDF files from other spacecraft as well, mainly via editing the `src/params.py` parameter file.)

~~The HPC version of the code currently ingests 300GB across 10,000 CDF files (data from 1995-2022) and produces an 18MB CSV file.~~

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

    HPC: 
         About 6000 files for the full 5 and a bit years of 6-hourly data from PSP = 130GB of disk space

4. **Delete files that are too small**

    `bash delete_small_files.sh`

    - A common error is that the file is unexpectedly small: i.e., it is not the full 6 hours, or at least does not contain 10,000 points at the initial resampled cadence. **We should run a simple check for this prior, deleting any files smaller than 1MB (and then seeing whether too many/not enough are caught by this, depending on any remaining error messages)**.

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
    - If some jobs do time out, meaning some files remain unprocessed, we can limit the `data/raw` directiory to those unprocessed files by moving them with `python move_matching_files.py`. *Maybe in future make this part of the slurm/python job*

    This script... **For PSP, there are an average of 4 files/file**

    See the notebook...

3. Perform train-test split for PSP data. **Be very careful about whether you want to delete existing files from the train-test folders. Currently these lines are commented out.**

    (make sure you have `module load`ed Python if on an HPC first)

    `python 2a_train_test_split.py`


4. **Compute the correction factor from all training set files**

    `python 2b_compute_heatmap.py`

    `sbatch 2b_compute_heatmap.sh`

    - 10 files, 15,20,25 bins **=1350 gapped ints = 5min 2GB**
    - 20 files '': 15min, 5GB **=2300 gapped ints = 9min 4GB**
    - **AVERAGE OF 5 INTERVALS PER FILE**

    - 7min, 2GB per 10 files.
    - For 200 files, = **=17,700 gapped ints = 135min, 32GB**
    - For 1000 files **latest results here**

5. **Perform the correction on the test set, file by file**

    Local: `for i in $(seq 0 5); do python 3_correct_test_sfs.py $spacecraft $i $n_bins; done`

    HPC: `sbatch 3_correct_test_sfs.sh`

    - 20min and 1GB/file (15,20,25 bins)

6. **Compute the statistical results for all (corrected) test set files**

    `python 4_compute_test_results.py $spacecraft $n_bins`

    Reqs: 

    - 10 files (wind) = **13x25 ints=18s, 0MB**
    - 20 files (Wind) = 4GB, 7min
    - 50 files (wind) = **68x25 ints = 12GB while running (WEIRD)**
    - 32-44 files (psp) = **47-90 ints = 5GB, 1min**


    43 files (PSP) = 12GB, 12min

    **Output: test_corrected_{spacecraft}_{bins}_bins.pkl** *not including ints, ints_gapped, sfs, or sfs_gapped_corrected*

7.  **Plot the test set results**
     (If on an HPC, download the above output at this step, as well as the heatmaps  and the **FIRST**  2-3 individual corrected pickle files for plotting case studies from) 
    `python 5a_plot_test_overall.py {spacecraft} {n_bins}`

    `python 5b_plot_test_case_studies.py  {spacecraft} {n_bins}`