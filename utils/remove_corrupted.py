import os

# Directory containing the files
directory = '/data_2/ShAPO_Data/CAMERA/train'

# List of files to be removed
import os

# Directory containing the files
directory = '/data_2/ShAPO_Data/CAMERA/train'

# List of files to be removed
files_to_remove = [
    "W3h7oQPkmES3E7CkQZeog•pickle.zstd",
    "nYmK5hTZmuSxPQ9q6hoVk.pickle.zstd",
    "bouiLWZXwwxkatxv8xxQVQ.pickle.zstd",
    "dv8yWSsNNZpzA2o7QDrN3n.pickle.zstd",
    "CQ5DfMFoEPV2kcVp8UBq• pickle.zstd",
    "GtxgJDsb4RXvNRCQUVH7oo.pickle.zstd"
]

# Iterate through the files to remove
for file_name in files_to_remove:
    file_path = os.path.join(directory, file_name)
    # Check if the file exists before attempting to remove it
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Removed {file_path}")
    else:
        print(f"File {file_path} does not exist")

print("Removal process completed.")

# Iterate through the files to remove
for file_name in files_to_remove:
    file_path = os.path.join(directory, file_name)
    # Check if the file exists before attempting to remove it
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Removed {file_path}")
    else:
        print(f"File {file_path} does not exist")

print("Removal process completed.")