dir=/nesi/nobackup/vuw04187/data/raw/psp
max_size=1000000  # 1 MB
#find $dir -maxdepth 1 -type f -size -"$max_size"c -delete

# check results 
find $dir -maxdepth 1 -type f -size -"$max_size"c -exec du -h {} \;
