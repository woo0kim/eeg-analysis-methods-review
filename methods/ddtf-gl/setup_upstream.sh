#!/bin/bash
# Unpack the authors' code bundle into methods/ddtf-gl/upstream/ (not committed: the bundle carries no license and is not
# published; it was shared with us as mwl_gl_si7_txt_bundle.zip, every file with an extra .txt suffix).
#
#   ./setup_upstream.sh <path to mwl_gl_si7_txt_bundle.zip>
#
# The README inside the bundle says to remove only the final .txt of every .py.txt / .yaml.txt / .yml.txt file and to keep
# the four library files in lib/; nothing else is changed. Our code imports lib/ and lnt_generate.py from here unchanged.
set -e
cd "$(dirname "$0")"
ZIP=${1:?usage: $0 <mwl_gl_si7_txt_bundle.zip>}
SHA=cd995bec08e2353a868cdf02ff93204d704f5b9a94d3b2fa024f2d8b315d0df6
if command -v sha256sum >/dev/null; then echo "$SHA  $ZIP" | sha256sum -c -
else echo "$SHA  $ZIP" | shasum -a 256 -c -; fi
mkdir -p upstream
python3 -I - "$ZIP" <<'EOF'
import os, sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
for n in z.namelist():
    if n.endswith('/'):
        continue
    rel = n.split('/', 1)[1]                                   # drop the share_mwl_gl_si7/ folder
    for ext in ('.py.txt', '.yaml.txt', '.yml.txt'):
        if rel.endswith(ext):
            rel = rel[:-4]
    out = os.path.join('upstream', rel)
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    open(out, 'wb').write(z.read(n))
    print('upstream/' + rel)
EOF
# lib/ has no __init__.py; Python imports it as a namespace package (lib.feat_utils), as the bundle's own scripts do.
