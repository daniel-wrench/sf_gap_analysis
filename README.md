# Gaps on structure functions: **interactive pipeline demo**
1. ~~Email Marcus~~
1. ~~Get heatmap importing and exporting~~
1. Plots
    - ~~MAPE in error trend line plots (means vs. means of means)? Using these in test set eval as well as determining bias initially?~~
    - ~~Add file indexing to later plots, move to appropriate locations.~~
    - ~~Get sample size plot (add *n* to heatmaps) Especially interested in how 3d heatmaps become more populated - *how often can test intervals not find the nearest bin?*~~
    - ~~Get plots to save. At earliest point in the code, whereever we're saving things, run the os func to check folder exists~~
1. ~~Run for 2 PSP ints for training, 1 PSP for testing.~~ See what dfs we are left with, and export the final ones.
1. ~~Run for 1 Wind testing~~
1. ~~Save and commit~~
2. ~~Move concatenation, as done as at start of step 3, from start of step 4 to start of step 5. Make step 4 run *on each test set file individually*.~~
1. ~~Divide each of the numbered steps into separate notebooks.~~ 
2. ~~Test these separate notebooks locally: 3 and 2 PSPs, 2 Wind~~
2. ~~Make quick to run on PSP vs. Wind, especially initial processing. Perhaps to just have if spacecraft =="psp": do this, elif "wind". Possibly easier to test this in scripts rather than notebooks.~~
2. ~~Quick run of each notebook~~
2. ~~Quick scroll through notebooks, checking for unnecessary outputs~~
3. ~~Convert notebooks to scripts.~~
    - ~~Note text output, make sure informative/concise enough.~~
    - ~~Turn file_index into a sys argument (this will be the job # on the HPC)~~
2. ~~Create new repo locally with necessary files~~
2. Full Python run on PSP
1. Push new repo to GitHub (don't need to commit data folders)
2. Speed up Wind data reading
2. Clone to NeSI, make job submission scripts
9. Quick scaling study on NeSI, investigating bad ints and possible automatic removal during download or initial reading
10. Results for 1 year of PSP, 1 month of Wind
10. Up the ante to two years, **while writing up existing results**. Maybe set aside geostats stuff for now.
11. Send manuscript to Tulasi, Fraternale, Marcus
12. Implement Fraternale's sample size threshold for fitting slopes

#### Notes
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

    Local: `for i in $(seq 0 5); do python 1_compute_sfs.py psp $i; done`

    HPC: `sbatch 1_compute_sfs.sh`
        
    - Recommended HPC job requirements: 256 cores/150 GB/7 hours (7-9min/file/core) for full 28-year dataset (works for 12, 8 and 4H interval lengths)
    
    This script processes magnetic field and velocity data measured in the solar wind by spacecraft to compute various metrics related to turbulent fluctuations and their statistical properties. It outputs the processed data for each input file into `data/processed/`.
        
    See the notebook **demo_scale_funcs.ipynb** for more on the numerical fitting. Fitting parameters, including the interval length, are specified in `params.py`. The most computationally expensive part of this script is the spectrum-smoothing algorithm, used to create a nice smooth spectrum for fitting slopes to.

4. **Compute the correction factor from all training set files**

5. **Perform the correction on the test set, file by file**

    Local: `for i in $(seq 0 5); do python 3_correct_test_sfs.py psp $i; done`

    HPC: `sbatch 3_correct_test_sfs.sh`

6. **Compute the statistical results for all (corrected) test set files**

    `python 4_compute_test_results.py psp'
