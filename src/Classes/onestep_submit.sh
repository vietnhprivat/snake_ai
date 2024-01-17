#!/bin/sh
#BSUB -J onestep17
#BSUB -o onestep17_output%J.out
#BSUB -e onestep17_error_ql%J.err
#BSUB -n 12
#BSUB -q gpua40
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=12G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 23:59
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
# end of BSUB options

nvidia-smi
# Load the cuda module
module load cuda/11.6

cd /zhome/db/e/206305/snake_ai
source intelsys/bin/activate
cd /zhome/db/e/206305/snake_ai/src/Classes
python agent_onestep.py