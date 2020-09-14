ws='/home/aming/MICRO/mobilenetv2/imagenet_pytorch_training'

for f in $ws/tuning/*/ImageNet_mobil*.stdout; do echo $f $(cat $f | grep "Early\|_Acc") | grep "Early\|_Acc" ; done
for f in $ws/tuning/*/ImageNet_mobil*.stderror; do echo $f $(cat $f | grep "Early\|_Acc") | grep "Early\|_Acc" | sed 's/print.*/early termination/g'; done
