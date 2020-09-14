
ws='/home/aming/MICRO/mobilenetv2/imagenet_pytorch_training'

RUN_NAME=$1 #ImageNet_mobileNetv2
FOLDER=$2
FILTER=$3 #'_1000000'
#cd $ws

RUNPATH=$ws/$FOLDER/tmps/${RUN_NAME}

mkdir -p $RUNPATH
for f in $RUNPATH*${FILTER}*
do
    #echo $f 
    NAME=$(basename $f)
    cat $f/checkpoints/log.txt | sed 's/\t/,/g' > $RUNPATH/${NAME}.csv
done

echo the first epoch
for f in $RUNPATH/*${FILTER}*; do printf "$f, "; echo " $(cat $f)" | head -2 | tail -1 ; done | sort -nr -k6 -t,
echo the top 10 best performings:
for f in $RUNPATH/*${FILTER}*; do printf "$f, "; echo " $(cat $f)" | tail -1; done | sort -nr -k6 -t, | head -10
rm $ws/$FOLDER/tmps/${RUN_NAME}/tuning_best.csv
cat $(for f in $RUNPATH/*${FILTER}*; do printf "$f, "; echo " $(cat $f)" | tail -1; done | sort -nr -k6 -t, | head -1 | sed 's/,.*//g') > $RUNPATH/tuning_best.csv

#python ../utils/choose_best.py "$RUNPATH"

