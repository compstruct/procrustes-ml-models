

source ~/.bashrc

ws='/home/aming/MICRO/mobilenetv2/imagenet_pytorch_training'

JOB_NAME=$(tail -1 <(squeueMe | grep $1) | sed 's/ /,/g' |sed 's/,,*/,/g' | cut -f3 -d',')
echo $JOB_NAME
#JOB_NAME='ImageNet_mobileNetv2@0.81_0.82_5_0.9_2000000_10_64_4e-05@tuning'
cat -n $ws/*/tmps/${JOB_NAME}/checkpoints/log.txt
