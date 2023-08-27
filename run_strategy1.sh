#!/bin/bash
# File: run.sh
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q all.q
#$ -N RunGA
#$ -M sp22078@essex.ac.uk
#$ -m be
#$ -o ~/log.out
#$ -pe smp 100

source activate sheehab_ga_sp22078
python ./run_ga_1.py

