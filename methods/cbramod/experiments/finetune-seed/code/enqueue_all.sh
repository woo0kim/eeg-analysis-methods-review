#!/bin/bash
# Every SEED / SEED-V pipeline run. The queue claims jobs in file-name order: cheap models first, then the models with
# a paper number to validate, then the slow ones. The subject-independent protocol runs the released / baseline
# implementation of each model; the braindecode twins ran under the trial split.
cd "$(dirname "$0")"
A="python3 add_jobs.py"
S="0 1 2 3 4"
for ds in SEED-V SEED; do
  for m in eegnet dgcnn bd-dgcnn bd-dgcnn-de bd-eegnet bd-deepsleepnet; do $A $ds trial555 10 $m $S; done
  for m in cbramod labram; do $A $ds trial555 20 $m $S; done
  for m in conformer bd-conformer; do $A $ds trial555 30 $m $S; done
  for m in bd-cbramod bd-labram cbramod-scratch; do $A $ds trial555 40 $m $S; done
  $A $ds trial555 50 bd-atcnet $S
  for m in eegnet dgcnn bd-deepsleepnet; do $A $ds subject 55 $m $S; done
  $A $ds subject 60 conformer $S
  for m in cbramod labram; do $A $ds subject 70 $m $S; done
  $A $ds subject 80 bd-atcnet $S
done
