import dataclasses
import operator
import time
import pathlib
import pickle
import shortuuid
import urllib3
# import boto3
import numpy as np
import zstandard as zstd
import os


def decompress_datapoint(cbuf):
  cctx = zstd.ZstdDecompressor()
  buf = cctx.decompress(cbuf)
  x = pickle.loads(buf)
  return x



data_path = '/data_2/ShAPO_Data/CAMERA/train'


all_files = os.listdir(data_path)

i=0
for file in all_files:
    
    if i >10000:
        print("Done with", i, "files")
    try:
        with open(file, 'rb') as fh:
            dp = decompress_datapoint(fh.read())
    except Exception as e:
        print(e)
        print("Error in file: ", file)
        continue
    i+=1