# Gaps on structure functions
*Furthering numerically improving our estimates of solar wind statistics (Re was implementing on large dataset, this is actually developing a new way)*

## Latest results

- Processed 3.5 years worth of PSP data (5000 6-hour files) = amount of clean data available from existing 5 and bit years of data, given current conditions; and 5 months of Wind data (same for all of 2016)
- Computed heatmap for 50 days worth of PSP data (200 files)
    - Average slope of 0.55, average tc of 1100s = 3km
    - Average intervals/file = 3.5
- Trained on 20,000 intervals, 25x785 PSP intervals (=2.5 months, coming from first 300 processed files), 15,20,25 bins, outdated slope range *takes about 3.5 hours*

![alt text](plots/temp/raapoi_test/test_wind_scatterplots_25_bins.png)

- Draft manuscript completed with these results; likely to be only minor updates with latest numbers and figs. Important to scale up analysis to populate heatmaps better, as well as using vector stats and better range (50-500 lags = 5-50% of $\lambda_C$)
- *NB: Previous slope range (1-10\% of corr length) did give results that matched theoretical values well, e.g. median of 0.67 from 175 PSP ints, 0.72 for 40 Wind ints*


## To-do
### Analysis

1. **~~Parallelise heatmap calculation.~~** 
    - **NEW APPROACH: Do by file instead, each core reads a number of files: saves the errors in a 20x20x20 array.**
    - Then when they're all merged in a serial job, we can calculate the mean, std, median, whatever we want, and get the correction intervals.
    - ~~Simplify heatmap part~~
    - ~~Simplify correction part~~
    - ~~Move existing plots~~
    - ~~Test new pipeline~~
    - ~~Run old pipeline on NESI subset on tiny subset~~
    - ~~Do all plots locally with proper Latex~~
    - ~~Note time and download stats, pull~~
    - ~~Run again, compare time and stats~~
    - ~~Make parallel, using gpt help: test locally, running for different ranges of lags then outputting these subranges, then merging them together again~~
    - ~~Test locally~~
    - ~~Push~~
    - ~~Run again, compare time and stats~~
4. **Choose #bins based on PSP data, report final results on Wind data**
4. Work on streamling 4b.py so we can manage the memory as best as possible (keeping in mind we kinda want to do it locally for latex anyway)
3. Potentially investigate smoothing and error bars - do they look OK as is?
11. Send completed draft manuscript to Tulasi. Print out, talk through with Marcus. Don't worry about Voyager just yet.3. Plot lag x-axes in units of $\lambda_C$?
3. Think about how to study Frat's method, and verify Burger's results
4. **Kurtosis** analysis
12. Implement Fraternale's sample size threshold for fitting slopes, and send to him
13. Read Ruzmaikin

### Manuscript and plots

*Chat with Tulasi when have final figures*

1. ~~Finish first draft of paper~~
2. Change colours of standardisation plot, make more obvious
3. Add in Wind interval #13 for case study, delete heatmap from corrected figure 
2. Depending on final results, prob remove slope APE from scatterplots, just have boxplots separately. Potentially make corrected taylor scale style plot
3. Get Latex error trend lines for subset of full results, still shows same pattern
2. Check Google Doc, Notion for notes, comments
3. Improve variogram clouds plot: what are we saying that isn't already covered by case study plots. And whatever that is, make it clear with good examples (probably same length as analysis intervals) and make the style consistent 
2. Make consistent (Latex) font, and specify sizes to match specifications in Overleaf

### Notes
- Calculate sf_2_pe in 1_compute_sfs? Currently not to have somewhat simpler calculation once corrected, but also leading to some duplication of code, especially if we want the error trend line plots.
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

    `python 2_train_test_split.py`


4. **Compute the correction factor from all training set files**

    Local:

    In `3_bin_errors.sh`, change `start_index` to 0 
    
    `bash 3_bin_errors.sh`

    HPC: 
    
    In `3_bin_errors.py`, adjust `data_prefix_path`, depending on where you are storing the data

    `sbatch 3_bin_errors.sh`

    - LATEST: 20 files/core, {2d, 3d} {15,20,25 bins} = 300MB, 3.5min
    - 5o files/core "" = CONSTANT 500MB, no matter how many files, 15s/file
    - Basically 15min to do the whole lot across 60 cores (73 files/core)

---

    
    - 10 files, 15,20,25 bins **=1350 gapped ints = 7min 3GB**
    - 20 files '': 15min, 5GB **=2300 gapped ints = 9min 4GB**
    - **AVERAGE OF 5 INTERVALS PER FILE**

    - ***10min, 3GB per 10 files (1750 gapped ints).***
    - ***PARALLEL: 5min, 3GB/core for 10 files, 5 cores***\
    - ***PARALLEL: 3min, 4GB/core for 15 files, 15 cores***


    - For 200 files, = **=17,700 gapped ints = 135min, 32GB**
    - For 1000 files **latest results here**

4. **Merge the binned errors and calculate the correction factor**

    Local: `bash 4a_finalise_correction.sh`

    HPC: `sbatch 4a_finalise_correction.sh`

    LATEST: 10 files (SERIAL JOB), {2d, 3d} {15,20,25 bins} = 200MB, 90s

    100 files "" = 350MB, 120s
    200 files "" = 500MB, 150s
    400 files "" = 820MB, 210s
    1000 () files "" = 1.7G, 7min


    - **4200 (all) files "" = 7G, 32min**

5. **Calculate the stats (average slope and corr time, error trend lines) for the training set** (not necessary for correction factor)

    `bash 4b_compute_training_stats.sh`

5. **Perform the correction on the test set, file by file**

    *111 Wind files to correct: outputs are 12-22KB each = 3MB files**

    THERE IS A VERSION FOR SAVING THE CORRECTED SFS, FOR COMPUTING OVERALL TEST RESULTS, AND A VERSION WITHOUT, FOR THE CASE STUDY PLOTS, TO BE RUN LOCALLY. NOTE ALSO DIFFERENT VERSIONS OF SF_FUNCS.LOAD_AND_CONCATENATE

    Local: `for i in $(seq 0 1); do python 5_correct_test_sfs.py $spacecraft $i $n_bins; done`

    HPC: `sbatch 5_correct_test_sfs.sh`

    **LATEST: 1GB AND 3MIN/FILE**
    ---

    - 20min and 1GB/file (15,20,25 bins)
    - **0-3min (prev 4-11) and 200-600MB/file (15,20)**

6. **Compute the statistical results for all (corrected) test set files**

    `bash/sbatch 6_compute_test_stats.sh`

    Reqs: 

    (Now using simplified outputs)
    - 30s and 100MB for the full 111 Wind files, containing 125 intervals


    43 files (PSP) = 12GB, 12min

    **Output: test_corrected_{spacecraft}_{bins}_bins.pkl** *not including ints, ints_gapped, sfs, or sfs_gapped_corrected*

7.  **Plot the test set results**
     (If on an HPC, download the above output at this step, as well as the heatmaps  and the **FIRST**  2-3 individual corrected pickle files for plotting case studies from) 
    `python 7a_plot_test_overall.py {spacecraft} {n_bins}`

    `python 7b_plot_test_case_studies.py  {spacecraft} {n_bins}`

    *For some reason this last one throws an error if running from the terminal, but is fine if running interactively*