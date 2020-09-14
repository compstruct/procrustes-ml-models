filter=$1
for f in ./${filter}*/tmps/*/checkpoints/log.txt
do
    fileName=$(echo $f | sed 's/.*tmps\///g' | sed 's/\//_/g')
    echo $fileName
    finalAcc=$(tail -1 $f | cut  -f 5 | sed 's/\..*//g')
    cp $f logs/${finalAcc}_${fileName}_$(hostname)
done
