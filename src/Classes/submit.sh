#!/bin/sh
#BSUB -J jobnam
#BSUB -o jobname%J.out
#BSUB -e jobname%J.err
#BSUB -n 4
#BSUB -q gpua10
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 8:00
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
# end of BSUB options

# module load cuda/11.8


cd /zhome/db/e/206305/snake_ai
source intelsys/bin/activate
cd /zhome/db/e/206305/snake_ai/src/Classes
python agent.py
