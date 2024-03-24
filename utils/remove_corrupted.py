import os

# Directory containing the files
directory = '/data_2/ShAPO_Data/CAMERA/train'

# List of files to be removed
files_to_remove = [
    "Sw2o3uFhAAGHw8XxnBRD9z.pickle.zstd",
    "EGh88y4QygcwcfqqaLqnxP.pickle.zstd",
    "ByyVgGMkf4VyfWvngTtLXN.pickle.zstd",
    "SY5RABqjS2V47eRZv6Kaq9.pickle.zstd",
    "SRKtE4Pd3tLNrXSBJA2S28.pickle.zstd",
    "UCD7WRWctHfha5iqbCwGwy.pickle.zstd",
    "AMByXeLRTngNhRzdhgBvcE.pickle.zstd",
    "iMrmLZrJ3cgpLVqJCXDSU8.pickle.zstd",
    "L8heQYcteF69TBpexFMwAL.pickle.zstd",
    "TKTvaxSAhahGzHN6owa6oY.pickle.zstd",
    "dv8yWSsNNZpzA207QDrN3n.pickle.zstd",
    "iDridYsR4BaTRRZ6dQQd4R.pickle.zstd",
    "CQ5DfMFoEPVV2kcVp8UBq.pickle.zstd",
    "n4mxcyQ6CobkCaUCNJW3QV.pickle.zstd",
    "KaQJNKWnc97xsGofZsDPL9.pickle.zstd",
    "7M6AtMmvtux6kgwFu9NcRR.pickle.zstd",
    "K4k3VMZTcVsyCVAEcPytGx.pickle.zstd",
    "GtxgJDsb4RXvNRCQUVH7o0.pickle.zstd",
    "Ed2S7TAPwBndptZfDoL5w8.pickle.zstd",
    "fX39uyUSa5ytBKKVZP4wJo.pickle.zstd",
    "UdjGR9SvsHiequMdg7ML7X.pickle.zstd",
    "jPHBrxYemfQEc2ZJHsHseC.pickle.zstd",
    "kde3ouVvDb5P7TceziJwBk.pickle.zstd",
    "W3h7oQPkmES3E7CkQZeog.pickle.zstd",
    "m9miQYPDgQek89fp29Y2qE.pickle.zstd",
    "fLjz62H28RgRCPSunpahjC.pickle.zstd",
    "ngjWZPqUAJwvXap2JkgJYX.pickle.zstd",
    "nYmK5hTZmuSxPQ9q6hoVk.pickle.zstd",
    "WK9mjFYHWxENteJYJwees2.pickle.zstd",
    "bouiLWZXwwxkatxv8xxQVQ.pickle.zstd",
    "ZTt6mem5BADM46uXvpQ5v8.pickle.zstd",
    "RmpX7vfzMnN5HegmrgR9xj.pickle.zstd",
    "T7nEp6wDEUSS4g6FpoVQ9A.pickle.zstd",
    "LeZkX4qZSejyfmbRBbeXu3.pickle.zstd",
    "ZDJsxHUnHwnbJKzJThRSDK.pickle.zstd",
    "VbM55wZRanrVAN3QXUA7Hh.pickle.zstd",
    "2fVAe4uqMPccMbTZqfGiot.pickle.zstd",
    "eh3mME3fLAW74KDgbqUKXR.pickle.zstd",
    "JA3D95r9rfN7NhP8uxYf6W.pickle.zstd",
    "YVfh8PxxEaggZdT27ywqFa.pickle.zstd",
    "a7nv4AahgWv6f8iQQp4AGa.pickle.zstd",
    "fRQnoNEgMF3WRWk7wR3hzk.pickle.zstd",
    "N5YjHN6omtwtcaBLcN8Tk7.pickle.zstd",
    "BXBrdgxr6sM8Wmb9RQP6KM.pickle.zstd",
    "2RN4SiZKuhhWqzY3KovFAd.pickle.zstd",
    "WKgqkLSzZiwFig5P7uHGcK.pickle.zstd",
    "Wtd3f2ZLsa45WkcaegLdo2.pickle.zstd",
    "RnX7gipmfALFeLLXzzLZbn.pickle.zstd"
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