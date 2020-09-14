
machines="jarvis1 jarvis2 jarvis3"
FILTER="'0.74_|0.2_|0.045|0.05'"
echo ',Epochs,Runs'
for m in $machines
do
	ssh $m "for runs in /home/aming/MICRO/mobilenetv2/imagenet_pytorch_training/*/tmps/ImageNet_*@*_fixedStep@*/checkpoints/log.txt; do wc -l" '$runs'" | sed 's/\/checkpoin.*//g' | sed 's/\/.*tmps\///g' | grep -E $FILTER | sed 's/  */ /g' | sed 's/ /,/g' | tr '\n' ' ' ; cat -n "'$runs'" | tail -1 ; done "
done

