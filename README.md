# Gaps on structure functions

## To-do

1. ~~Commit changes!~~
2. LOCALLY
    - ~~Change scatterplots to ANN paper version~~
    - ~~Change correction interval to pm 2SD (NOT se)~~
    - ~~Change boxplots to violin plots, if easily done for seaborn. If not, just include outliers in tails~~
    - ~~Get mean slopes from true SFs~~
3. ~~Revert back to HPC params and push~~
4. ~~Make nice improvements to paper intro~~
5. ~~**Get full results on subset of data**~~
    - Trained on 216 processed PSP files (171=43 days, coming from first 400 raw files)
    - Tested on some bin ranges on 44 PSP (11 days) and 20 Wind files (20 days).
    - Old slope range (10-100)
    - Old bin range (10,15,20) 
6. ~~Process all files and train-test split (2019-2020) **using old slope range** = 1159 training files~~
6. Test out computing bigger bins on larger subset of training files: 15,20,25 on first 400
    - ~~Trained on 300 processed PSP files (5 months)~~
    - ~~Tested on PSP test set with all 3 bin sizes~~
    - ~~Tested on Wind test set with all 3 bin sizes~~
    - ~~Calculate results for Wind dataset with all 3 bin sizes (20 files)~~
    - For PSP (100 files)
6. Update scatterplot (25 versions of 20 Wind files) 
    - **Speed up plot iteration process** (get the latest scatterplot to show at meeting, plus an example)
    - Make statement about when to use which method
    - Performance on PSP?
7. Calculate correlation between slope APE and MAPE
7. Run remainder of pipeline **using old slope range** and equivalently sized test sets.
7. Space out files in step 1. Look at other notes below to see if any easy things to knock over.
8. With the latest numbers for job requirements, and bigger bin results, re-run full pipeline on all data with new slope range.

5. Meanwhile, new manuscript, just taking intro/bg from existing and not doing geostats stuff for now.
    - Get latest best plots from NESI, chuck em in, and GET WRITING! (Visual editor)
11. Send completed draft manuscript to Tulasi, Marcus. Don't worry about Voyager just yet.
12. Implement Fraternale's sample size threshold for fitting slopes, and send to him

### Notes
- Processed PSP and Wind files are between 32 and 156MB each
- Add true SFs to case study plots. Will require re-jig of `ints` df
- Get mean slopes from true SFs. Maybe move "inertial range" if consistent bias wrt 2/3
- Problem with first step is uneven times due to some files having no intervals, some having up to 4. Might be better to run on 3-5 files, spaced out (i.e. every 3rd file) in order to get more even times across jobs.
- Would be nice to get total # intervals for each set returned by step 1
- investigating bad ints and possible automatic removal during download or initial reading
- consistency between times_to_gap across files
- Wind data reads very slowly, compared with PSP. It is using a pipeline function that I think Kevin made, made up of many smaller functions.
The bottleneck is the "format epochs" function. I've starting trying to do this in the same was as PSP, but it was struggling to do the timedelta addition
- Can add smoothing to correction step alter, **not on critical path for getting most of the scripts on NESI**
- Having logarithmically spaced lag bins would make the correction factor much cleaner to work with: one-to-one bins
- For now likely to do stick with simple job arrays and single jobs on HPC, with importing and exporting of intermediate steps, but perhaps better to do single MPI script with broadcasting and reducing.
- If we tend to just get one 1 interval out of each file, could do away with (file,int) indexing
- In either case, might using actual pandas indexes be easier/more efficient?
- Add sf slope to Wind dataset

## How to run this code

(It should be relatively easy to adjust to use CDF files from other spacecraft as well, mainly via editing the `src/params.py` parameter file.)

The HPC version of the code currently ingests 300GB across 10,000 CDF files (data from 1995-2022) and produces an 18MB CSV file.

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

    HPC: 
         There are approximately 10,000 files for each of the three daily datasets. **This results in a requirement of 300GB of disk space**.
    
    - (`tmux new`) (this allows you to work in another session while the files are downloading)
    - `srun --pty --cpus-per-task=2 --mem=1G --time=02:00:00 --partition=quicktest bash`
    - `bash 0_download_files.sh`
    - (`Ctrl-b d` to detach from session, `tmux attach` to re-attach)

4. **Process the data, file by file**

    Local:

    Adjust data_prefix_path, depending on where you are storing the data (likely in code dir, so set to `""`)

    `for i in $(seq 0 5); do python 1_compute_sfs.py $spacecraft $i; done`

    HPC: 
    
    Adjust `data_prefix_path`, depending on where you are storing the data


    `sbatch 1_compute_sfs.sh`
        
    - For 1 file each, job reqs of up to 1GB and rarely over 15min time (often much less when no good intervals found)
        PSP: From complete set of 400 raw files, we got 216 output files containing good intervals (with just the occassional timeout) **FEBRUARY 2019 IS NO GOOD (can tell from looking at plots/temp/, where only acfs are plotted**
        Wind: From 30 tries, we got 20 good files.
    This script processes magnetic field and velocity data measured in the solar wind by spacecraft to compute various metrics related to turbulent fluctuations and their statistical properties. It outputs the processed data for each input file into `data/processed/`.
        
    See the notebook **demo_scale_funcs.ipynb** for more on the numerical fitting. Fitting parameters, including the interval length, are specified in `params.py`. The most computationally expensive part of this script is the spectrum-smoothing algorithm, used to create a nice smooth spectrum for fitting slopes to.

3. Perform train-test split for PSP data

    (make sure you have `module load`ed Python if on an HPC first)

    `python 2a_train_test_split.py`

4. **Compute the correction factor from all training set files**

    `python 2b_compute_heatmap.py`

    `sbatch 2b_compute_heatmap.sh`

    HPC job: for 10 files, 3 sets of bins: 2.5min, 1.5GB
    20 files '': 5min, 3GB
    170 files '': 45min, 25GB

    10 files, 15,20,25 bins: 5min, 1.7GB
    30 files '': 15min, 5GB
    300 files '': 
    400 files, '': >6 hours, >72GB
    

5. **Perform the correction on the test set, file by file**

    Local: `for i in $(seq 0 5); do python 3_correct_test_sfs.py $spacecraft $i $n_bins; done`

    HPC: `sbatch 3_correct_test_sfs.sh`

6. **Compute the statistical results for all (corrected) test set files**

    `python 4_compute_test_results.py $spacecraft $n_bins`

    Reqs: 
    
    20 files (Wind) = 4GB, 7min

    43 files (PSP) = 12GB, 12min