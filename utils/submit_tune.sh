CONFIG=$1
BENCH=$2
FOLDER=$3
GPUs=$4
#WORKSPACE=$4
#RESULT=$5

#BENCH='ImageNet_mobileNetv2'
#CONFIG='0.9_0.0125_1000000' # DC_LR_OT
#FOLDER='tuning'

ws='/home/aming/MICRO/mobilenetv2/imagenet_pytorch_training'
rs='/home/aming/MICRO/mobilenetv2/imagenet_pytorch_training'

cd $rs
mkdir -p $FOLDER
mkdir -p $FOLDER/$CONFIG
cd $ws

STDOUT="$rs/${FOLDER}/${CONFIG}/${BENCH}_%A_%a.stdout"
STDERR="$rs/${FOLDER}/${CONFIG}/${BENCH}_%A_%a.stderror"

SLURM_JOB_NAME="${BENCH}@${CONFIG}@${FOLDER}"

machineName='beluga' # befor submitting!

if [[ 5 < $(expr match $(hostname)  ${machineName} ) ]];
then
       echo "This machine is: $machineName: $(hostname)"
       sbatch --output=$STDOUT --error=$STDERR -J ${SLURM_JOB_NAME}  $ws/RunBench_beluga.slurm

elif [[ 4 < $(expr match $(hostname)  'cedar' )  ]]
then
       echo "This machine is: cedar: $(hostname)"
       cd /scratch;
       sbatch --output=$STDOUT --error=$STDERR -J ${SLURM_JOB_NAME}  $ws/RunBench_cedar.slurm
       cd $ws
else
       if [[ $(hostname) == 'jarvis2'  ]]
       then
           echo "This machine is Jarvis2: $(hostname) - running on ${GPUs} GPUs"
           sbatch --output=$STDOUT --error=$STDERR --time=70:00:00 -p vip -J ${SLURM_JOB_NAME}  $ws/RunBench_jarvis_${GPUs}.slurm
       else
           echo "This machine is other Jarvis for now: $(hostname) - running on ${GPUs} GPUs"
           sbatch --output=$STDOUT --error=$STDERR --time=48:00:00 -J ${SLURM_JOB_NAME}  $ws/RunBench_jarvis_${GPUs}.slurm
       fi

fi

