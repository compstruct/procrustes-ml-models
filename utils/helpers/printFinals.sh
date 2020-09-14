machines="jarvis1 jarvis2 jarvis3"
models='efficient resnet18 mobileNet'
filters='_False _2000000 _1000000'
for filter in $filters
do
for model in $models
do
    command="cd ~/MICRO/mobilenetv2/imagenet_pytorch_training/; bash helpers/final_tuning_vis.sh | grep $model | grep $filter | tail -1"
    for m in $machines
    do
        ssh $m "$command"
    done
done
done
