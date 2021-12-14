# https://github.com/entitize/Fakeddit

import argparse
import pandas as pd
import os
from tqdm import tqdm as tqdm
import urllib.request
import numpy as np
import sys

parser = argparse.ArgumentParser(description='r/Fakeddit image downloader')

parser.add_argument('type', type=str, help='train, validate, or test')

args = parser.parse_args()

df = pd.read_csv(args.type, sep="\t")
df = df.replace(np.nan, '', regex=True)
df.fillna('', inplace=True)

pbar = tqdm(total=len(df))

num_failed = 0

if not os.path.exists("images"):
  os.makedirs("images")
for index, row in df.iterrows():
  if row["hasImage"] == True and row["image_url"] != "" and row["image_url"] != "nan":
    image_url = row["image_url"]
    try:
      image_path = os.path.join("images", row["id"] + ".jpg")
      if not os.path.exists(image_path):
        urllib.request.urlretrieve(image_url, "images/" + row["id"] + ".jpg")
    except:
      num_failed += 1
  pbar.update(1)
print("done")
print("num_failed: {}".format(num_failed))
