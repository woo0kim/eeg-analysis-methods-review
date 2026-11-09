#!/bin/bash
# Fetch the DGCNN authors' code into methods/dgcnn/upstream/ (not committed: the archive carries no license).
# Source: the [code] link next to the paper on Tengfei Song's page (http://aip.seu.edu.cn/songtengfei). The page
# no longer serves it, so this reads the Wayback Machine's capture of 2022-11-22. braindecode's dgcnn.py cites the
# same archive as the official code. Only .py sources are extracted (the zip also holds .pyc, .swp and backups).
set -e
cd "$(dirname "$0")"
URL=http://web.archive.org/web/20221122130132id_/http://aip.seu.edu.cn/wp-content/uploads/2021/08/EEG-TAC.zip
SHA=15e4df187641b138261935aa05ff08e73fdb4022865ff7f1cbe630f44d1e268f
mkdir -p upstream
[ -f upstream/EEG-TAC.zip ] || curl -sSL -o upstream/EEG-TAC.zip "$URL"
if command -v sha256sum >/dev/null; then echo "$SHA  upstream/EEG-TAC.zip" | sha256sum -c -
else echo "$SHA  upstream/EEG-TAC.zip" | shasum -a 256 -c -; fi
python3 -I -c "
import zipfile
z = zipfile.ZipFile('upstream/EEG-TAC.zip')
for n in z.namelist():
    if n.startswith('EEG-TAC/') and n.endswith('.py'):
        z.extract(n, 'upstream')
print('extracted to methods/dgcnn/upstream/EEG-TAC')
"
