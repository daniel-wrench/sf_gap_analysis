module load python/3.10.5-gimkl-2022a
source venv/bin/activate
# If running (locally) on Windows, may need to change above lines to the following: 
#source venv/Scripts/activate

echo "JOB STARTED"
date
run_mode=full

# spacecraft="wind"
# echo "SPACECRAFT: $spacecraft"

python 6_compute_test_stats.py wind > results/$run_mode/test_stats.txt

echo "JOB FINISHED"
date
