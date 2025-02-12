import os

def move_matching_files(source_dir1, source_dir2, target_dir):
    """Moves files with matching names and extensions from source directories to target."""

    # Get a list of filenames from both source directories
    files1 = os.listdir(source_dir1)
    files2 = os.listdir(source_dir2)

    # Find matching filenames and extensions and move them
    for file in files1:
        base_name, ext = os.path.splitext(file)
        if base_name in [os.path.splitext(f)[0] for f in files2]:
            source_path = os.path.join(source_dir1, file)
            target_path = os.path.join(target_dir, file)
            os.rename(source_path, target_path)
            print(f"Moved {file} to {target_dir}")

# Example usage:
source_dir1 = "/nesi/nobackup/vuw04187/data/raw/psp/"
source_dir2 = "/nesi/nobackup/vuw04187/data/processed/psp/test/"
target_dir = "/nesi/nobackup/vuw04187/data/raw/psp/done/"

move_matching_files(source_dir1, source_dir2, target_dir)
