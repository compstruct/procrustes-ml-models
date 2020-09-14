#!/bin/bash
#SBATCH --mem=16gb
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=1-10%1
#SBATCH --gres=gpu:1
#SBATCH --account=def-mieszko

ws='/home/aming/MICRO/mobilenetv2/imagenet_pytorch_training'

echo "started the RunBench at $(date)"
module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torch torchvision --no-index
pip install matplotlib tensorboardX --no-index
pip install pandas numpy --no-index
pip install progress
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali

# compile the QE code under utils/torch_qe
pip install $ws/utils/torch_qe/setup.py

# ---- INIT PATHS ---------------

#ws='/home/aming/MICRO/mobilenetv2/imagenet_pytorch_training'
rs=$(cat $ws/rs.path)
sc=$rs
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

# ----- PATHS -----------------------
#INPUT_FOLDER=$ws/${FOLDER}/${CONFIG}
#OUTPUT_FOLDER=$rs/${FOLDER}/${CONFIG}

cd $SCRATCH
mkdir -p $FOLDER; cd $FOLDER
mkdir -p tmps; cd tmps
mkdir -p $SLURM_JOB_NAME; cd $SLURM_JOB_NAME

tmpPath=$SCRATCH/$FOLDER/tmps/$SLURM_JOB_NAME
cp $ws/schedules/${SLURM_JOB_NAME}.csv $tmpPath/schedule.csv
#SING_IMG='DB0220.simg'
IN1='/project/def-mieszko/IMAGENET-UNCROPPED' #'/scratch/aming/IMAGENET-UNCROPPED'
echo started copying to $SLURM_TMPDIR at: $(date) 
mkdir $SLURM_TMPDIR/IMAGENET-UNCROPPED
mkdir $SLURM_TMPDIR/IMAGENET-UNCROPPED/train
mkdir $SLURM_TMPDIR/IMAGENET-UNCROPPED/val
#cp -r $IN1 $SLURM_TMPDIR

avail=$(df --output=avail $SLURM_TMPDIR | tail -1)
need=$(echo '130 * 1024 * 1024' | bc)

if [[ $avail > $need ]]
then

    echo 'Enough space on localscratch!'

    ls  -1 /project/def-mieszko/IMAGENET-UNCROPPED/val   | xargs -n1 -P16 -I% rsync --info=progress2 --chown=aming:def-mieszko -r /project/def-mieszko/IMAGENET-UNCROPPED/val/%   $SLURM_TMPDIR/IMAGENET-UNCROPPED/val
    echo Number of copied val files: $(ls -R  $SLURM_TMPDIR/IMAGENET-UNCROPPED/ | wc -l)
    echo finished copying val at: $(date)
    ls  -1 /project/def-mieszko/IMAGENET-UNCROPPED/train | xargs -n1 -P16 -I% rsync --info=progress2 --chown=aming:def-mieszko -r /project/def-mieszko/IMAGENET-UNCROPPED/train/% $SLURM_TMPDIR/IMAGENET-UNCROPPED/train

    copied=$(ls -R  $SLURM_TMPDIR/IMAGENET-UNCROPPED/ | wc -l)
    echo Number of copied files total : $copied
    echo finished copying at: $(date)

    if [[ $copied > 1330000 ]]
    then
        echo properly copied!

        IN=$SLURM_TMPDIR/IMAGENET-UNCROPPED # run on local node => faster speed
        echo "This machine is cedar: $(hostname)"
        # ------ RUN INSIDE SINGULARITY IMAGE -------------------
        #singularity exec --nv -B /home -B /scratch /home/aming/${SING_IMG} sh -c "sleep 10; nvidia-smi; cd $tmpPath; echo running at:$(pwd); bash $ws/run_tune.sh $LR $LD $LS $ID $TS $EP $BS $WD $MO $DB $ST $QI $QS $QF $MD $IN"
        nvidia-smi; cd $tmpPath; echo running at:$(pwd); bash $ws/run_tune.sh $LR $LD $LS $ID $TS $EP $BS $WD $MO $DB $ST $QI $QS $QF $MD $IN

    else
        echo not enough space or dead rsync or bad copy!
    fi

else
    echo 'not enough space on localscratch'
    echo trying to find a prev copy!
    PREV_COPY=$(bash $ws/check_node.sh )
    if [[ '' !=  $PREV_COPY ]]
    then 
        echo found prev copy here: $PREV_COPY; 
        IN=$PREV_COPY
        nvidia-smi; cd $tmpPath; echo running at:$(pwd); bash $ws/run_tune.sh $LR $LD $LS $ID $TS $EP $BS $WD $MO $DB $ST $QI $QS $QF $MD $IN
    else
        echo no copy no space! Wth!; 
    fi
fi
