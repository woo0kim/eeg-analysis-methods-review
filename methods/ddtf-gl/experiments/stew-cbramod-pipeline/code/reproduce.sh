#!/bin/bash
# dDTF-GL, stew-cbramod-pipeline. The code is shared by every dDTF-GL experiment and lives in methods/ddtf-gl/code (README: ../../README.md).
#   code/reproduce.sh collect   recompute results/ from logs/runs into results_rerun/ and diff (numpy, scikit-learn)
#   code/reproduce.sh train     features and every run (CPU; needs the data, upstream/ and the ddtf-gl environment)
exec "$(cd "$(dirname "$0")" && pwd)/../../../code/reproduce.sh" stew-cbramod-pipeline "$@"
