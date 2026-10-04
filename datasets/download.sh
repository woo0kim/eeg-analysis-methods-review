#!/bin/bash
# Download one public EEG dataset into $EEG_DATA_ROOT (default ~/data). Needs wget, curl and GNU coreutils.
# usage: datasets/download.sh <mental|physio|bciciv2a|mumtaz|shu>
D=${EEG_DATA_ROOT:-~/data}
S3=https://physionet-open.s3.amazonaws.com   # official PhysioNet open-data mirror (AWS)
case "$1" in
mental)   # MentalArithmetic = PhysioNet "eegmat" 1.0.0
  mkdir -p $D/mental_arithmetic/edf && cd $D/mental_arithmetic
  wget -q $S3/eegmat/1.0.0/SHA256SUMS.txt -O SHA256SUMS.txt
  grep "\.edf$" SHA256SUMS.txt | awk '{print $2}' | xargs -P 16 -I{} wget -q -c $S3/eegmat/1.0.0/{} -O edf/{}
  (cd edf && grep "\.edf$" ../SHA256SUMS.txt | sha256sum -c --quiet && echo "sha256 OK")
  ls edf | wc -l ;;
physio)   # PhysioNet-MI = PhysioNet "eegmmidb" 1.0.0 (imagery runs used by preprocessing_physio.py)
  mkdir -p $D/physio/files && cd $D/physio
  wget -q $S3/eegmmidb/1.0.0/SHA256SUMS.txt -O SHA256SUMS.txt
  grep -E "R(04|06|08|10|12|14)\.edf$" SHA256SUMS.txt > needed.sha256
  awk '{print $2}' needed.sha256 | xargs -P 32 -I{} sh -c 'mkdir -p files/$(dirname {}) && wget -q -c '$S3'/eegmmidb/1.0.0/{} -O files/{}'
  (cd files && sha256sum -c --quiet ../needed.sha256 && echo "sha256 OK")
  ls files | wc -l; find files -name "*.edf" | wc -l ;;
bciciv2a) # BCI Competition IV-2a, BNCI Horizon 2020 data set 001-2014
  mkdir -p $D/bciciv2a/data_mat && cd $D/bciciv2a/data_mat
  for s in 1 2 3 4 5 6 7 8 9; do for t in T E; do echo A0${s}${t}.mat; done; done | xargs -P 6 -I{} wget -q -c https://bnci-horizon-2020.eu/database/data-sets/001-2014/{}
  ls | wc -l ;;
mumtaz)   # Mumtaz2016, figshare article 4244171
  mkdir -p $D/mumtaz/files && cd $D/mumtaz/files
  curl -sS "https://api.figshare.com/v2/articles/4244171/files?page_size=1000" > ../filelist.json
  python3 - <<"PY"
import json, subprocess, os, collections, hashlib
fl = json.load(open("../filelist.json"))
cnt = collections.Counter(f["name"] for f in fl)
bad = []
for f in fl:
    # figshare "download all" prefixes duplicated file names with the file id (cf. CBraMod issue #13)
    name = f["name"] if cnt[f["name"]] == 1 else "%d_%s" % (f["id"], f["name"])
    if not os.path.exists(name) or os.path.getsize(name) != f["size"]:
        subprocess.run(["wget", "-q", "-c", f["download_url"], "-O", name], check=False)
    if hashlib.md5(open(name, "rb").read()).hexdigest() != f.get("computed_md5"):
        bad.append(name)
print("mumtaz files:", len(os.listdir(".")), "| duplicated names:", [k for k, v in cnt.items() if v > 1], "| md5 mismatches:", bad)
PY
  ;;
shu)      # SHU-MI, figshare article 19228725 (mat_files.zip)
  mkdir -p $D/shu && cd $D/shu
  wget -q -c https://ndownloader.figshare.com/files/36728994 -O mat_files.zip
  echo "6c039cce4025b2749545949c93f7a4f1  mat_files.zip" | md5sum -c
  unzip -q -o mat_files.zip -d mat_unzip
  find mat_unzip -name "*.mat" | wc -l ;;
esac
echo "DONE $1"
