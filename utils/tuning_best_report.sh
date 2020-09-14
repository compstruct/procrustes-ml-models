
RUN_NAME=$1 #ImageNet_mobileNetv2
FILTER=$2 #'_1000000'
#cd $ws

mkdir -p tuning/tmps/${RUN_NAME}
for f in tuning/tmps/${RUN_NAME}*${FILTER}*
do
    #echo $f 
    NAME=$(basename $f)
    cat $f/checkpoints/log.txt | sed 's/\t/,/g' > tuning/tmps/$RUN_NAME/${NAME}.csv
done

echo the first epoch
for f in ./tuning/tmps/${RUN_NAME}/*${FILTER}*; do printf "$f, "; echo " $(cat $f)" | head -2 | tail -1 ; done | sort -nr -k6 -t,
echo the top 10 best performings:
for f in ./tuning/tmps/${RUN_NAME}/*${FILTER}*; do printf "$f, "; echo " $(cat $f)" | tail -1; done | sort -nr -k6 -t, | head -10

#python choose_best.py "tuning/tmps/${RUN_NAME}"

