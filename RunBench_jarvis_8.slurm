#!/bin/bash
#SBATCH --mem=192gb
#SBATCH --array=1-10%1
#SBATCH --cpus-per-task=32
#SBATCH --mail-user=aming@ece.ubc.ca
#SBATCH --gres=gpu:8

# ---- INIT PATHS ---------------

ws='/home/aming/MICRO/mobilenetv2/imagenet_pytorch_training'
rs='/home/aming/MICRO/mobilenetv2/imagenet_pytorch_training'
sc=$ws
SCRATCH=$sc #/home/aming/scratch

#------- PARAMETERS --------------
FOLDER="${SLURM_JOB_NAME##*@}"
REMAINING="${SLURM_JOB_NAME%@*}"
BASENAME="${REMAINING%%@*}"
BENCH="${REMAINING%@*}"
CONFIG="${REMAINING##*@}"
#echo $FOLDER
#echo $BASENAME
#echo $BENCH
#echo config $CONFIG

#----- BENCH ------------------
REMAINING=${BENCH}
#1
IN_DUMMY="${REMAINING%%_*}"
REMAINING="${REMAINING#*_}"
#2
MD="${REMAINING%%_*}"
REMAINING="${REMAINING#*_}"

#echo $MD

#----- CONFIGS ------------------
REMAINING=${CONFIG}
#1
LR="${REMAINING%%_*}"
REMAINING="${REMAINING#*_}"
#2
LD="${REMAINING%%_*}"
REMAINING="${REMAINING#*_}"
#3
LS="${REMAINING%%_*}"
REMAINING="${REMAINING#*_}"
#4
ID="${REMAINING%%_*}"
REMAINING="${REMAINING#*_}"
#5
TS="${REMAINING%%_*}"
REMAINING="${REMAINING#*_}"
#6
EP="${REMAINING%%_*}"
REMAINING="${REMAINING#*_}"
#7
BS="${REMAINING%%_*}"
REMAINING="${REMAINING#*_}"
#8
WD="${REMAINING%%_*}"
REMAINING="${REMAINING#*_}"
#9
MO="${REMAINING%%_*}"
REMAINING="${REMAINING#*_}"
#10
DB="${REMAINING%%_*}"
REMAINING="${REMAINING#*_}"
#11
ST="${REMAINING%%_*}"
REMAINING="${REMAINING#*_}"

#12
QI="${REMAINING%%_*}"
REMAINING="${REMAINING#*_}"
#13
QS="${REMAINING%%_*}"
REMAINING="${REMAINING#*_}"
#14
QF="${REMAINING%%_*}"
REMAINING="${REMAINING#*_}"

#echo $LR
#echo $LD
#echo $LS
#echo $ID
#echo $TS
#echo $EP
#echo $BS
#echo $WD
#echo $MO
#echo $DB
#echo $ST
#echo $QI
#echo $QS
#echo $QF

# ----- PATHS -----------------------
#INPUT_FOLDER=$ws/${FOLDER}/${CONFIG}
#OUTPUT_FOLDER=$rs/${FOLDER}/${CONFIG}

cd $SCRATCH
mkdir -p $FOLDER; cd $FOLDER
mkdir -p tmps; cd tmps
mkdir -p $SLURM_JOB_NAME; cd $SLURM_JOB_NAME

tmpPath=$SCRATCH/$FOLDER/tmps/$SLURM_JOB_NAME
cp $ws/schedules/${SLURM_JOB_NAME}.csv $tmpPath/schedule.csv
SING_IMG='qe_0516.simg'
IN='/datasets/IMAGENET-UNCROPPED'
echo "This machine is Jarvis (for now): $(hostname)"
# ------ RUN INSIDE SINGULARITY IMAGE -------------------
singularity exec --nv -B /home -B /scratch -B /datasets /home/aming/${SING_IMG} sh -c "sleep 10; nvidia-smi; cd $tmpPath; mkdir -p dumps; echo running at:$(pwd); bash $ws/run_tune.sh $LR $LD $LS $ID $TS $EP $BS $WD $MO $DB $ST $QI $QS $QF $MD $IN"

