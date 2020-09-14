
echo $1 #dump_test/tmps/ImageNet_resnet18@0.4_0_1_0.9_2000000_100_32_4e-05_0_True_plateau2@dump_test/
#JB=$(echo $1 | grep -o '/.*/' | sed 's/\///g' |sed 's/tmps//g') #ImageNet_resnet18@0.4_0_1_0.9_2000000_100_32_4e-05_0_True_plateau2@dump_test
JB=$(echo $1 | sed 's/.*\///g')
echo JB $JB
JB_BASENAME=$(echo $JB | grep -o '.*@')
echo JB_BASENAME $JB_BASENAME

#dumps/resnet18_dp-True_wt-2000000_ep-70_itt-0_summary_sparsity.txt
cp -r $1 dump_test/tmps/${JB_BASENAME}dump_test
echo CP to dump_test/tmps/${JB_BASENAME}dump_test

export SLURM_JOB_NAME=${JB_BASENAME}dump_test
echo $SLURM_JOB_NAME
source RunBench_jarvis_test.slurm

for f in  dumps/resnet18_dp-T*summary_sparsity.txt
do
    cat $f | grep -v 'bn\|downsample_1' > $f.csv
done

python /home/aming/MICRO/mobilenetv2/imagenet_pytorch_training/plot_bar.py $f.csv $SLURM_JOB_NAME
scp $f.csv.pdf rigel:~/pdfs_sparsity

cd /home/aming/MICRO/mobilenetv2/imagenet_pytorch_training
