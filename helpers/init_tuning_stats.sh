
for f in */tmps/*_*; do echo "$(wc -l $f/checkpoints/log.txt | cut -f1 -d' ' ), $(cat $f/checkpoints/log.txt | head -2 | tail -1| cut -f 5), $(cat $f/checkpoints/log.txt | head -2 | tail -1| cut -f 1) , $f" | sort | grep -v Valid | grep -v ',  ,' ; done
