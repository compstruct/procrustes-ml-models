
#----------------------------------------------------------

MODEL=$1 #'mobilenetv2'
INPUT='ImageNet'
ID=$2 # '_tuning'
FOLDER=$3 #'tuning'/'baseline'

ws='/home/aming/MICRO/mobilenetv2/imagenet_pytorch_training'
rs='/home/aming/MICRO/mobilenetv2/imagenet_pytorch_training'

#----------------------------------------------------------
BENCH=${INPUT}_${MODEL} #'ImageNet_mobileNetv2'

mkdir -p $FOLDER
mkdir -p $FOLDER/tmps
mkdir -p $FOLDER/tmps/$BENCH
mkdir -p $FOLDER/logs/$BENCH

# find the best performing in past
FILTER='' #'_2000000'
bash helpers/tuning_best.sh $BENCH $FOLDER $FILTER 

now=$(echo $(date) | sed 's/ /_/g' | sed 's/:/_/g')
python tuning.py $MODEL $ID > tuning.log # outputs to tuning.runs
# just for profiling
cp tuning.log  $FOLDER/logs/$BENCH/tuning_${now}.log
cp tuning.runs $FOLDER/logs/$BENCH/tuning_${now}.runs

GPUs=1
#Submit jobs randomly
shuf tuning.runs > tuning.runs.shuf
while IFS= read -r CONFIG
do
    echo "submitting $CONFIG $BENCH $FOLDER"
    bash submit_tune.sh $CONFIG $BENCH $FOLDER $GPUs
done < tuning.runs.shuf

#----------------------------------------------------------
