# Neural Network Training Management Framework with PyTorch 

![Tuning Cycle](./resources/tuningCycle.png)

The training script is edited from reproduction of MobileNet V2 architecture as described in [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)

<!---
 | Table of Contents |  |
 |-------------------|--|
 | Tuning            |  |
 | Things to set     |  |
 | Helper functions  |  |
 | Parameter counts  |  |
 | Dataset           |  |
 | Clusters          |  |
 | File transfer     |  |
--->

## Requirements
 + Install PyTorch (pytorch.org)
 + pip install -r requirements.txt
 + Download the ImageNet dataset from http://www.image-net.org/

## Tuning
For training, introduce your model parameters in models/parameters.json and call it similar to examples in `runOnCluster.sh`

Hierarchy:

```
runOnCluster.sh [model_name, sub-conf, out_dir]
└── tune.sh > tuning.log
    ├── tuning.py [*model_name] > tuning.runs
    └── submit_tune.sh
        └── RunBench[_cedar/_beluga/_local].slurm
            └── run_tune.sh (inside Singularity)
                └── imagenet_train.py [generated_tuning_arguments]
```

The output of each training is at: `out_dir/_config_/_runName_jobArrayId.std[out/error]`
The log of each training is at: `out_dir/tmps/_run_name@_config_@_out-dir_/checkpoints/log.txt`

Each RunBench have diff groups, partition, scheduling time, singularity dir binding.
the submit_tune.sh script will automatically figure out which cluster you are using now by simple pattern matching on the hostname.

 + Local: RunBench_local_1/8.slurm: 1 or 8 GPUs on local clusters
 + Compute Canada: RunBench_beluga.slurm and RunBench_cedar.slurm. no vip partition
  
 

The job name that is passed to RunBench.slurm has 3 field separated by '@': e.g. `ImageNet_mobileNetv2@0.0625_0.25_2_0.9_2000000_10_64_4e-05_0_False_fixedStep@tuning`
config is passed like this: `0.0625_0.18_2_0.9_2000000_10_64_4e-05_0_True_fixedStep

e.g. output: `tuning/0.0625_0.18_2_0.9_2000000_10_64_4e-05_0_False_fixedStep/ImageNet_mobileNetv2_2201_1.stdout`
e.g. log: `tuning/tmps/ImageNet_mobileNetv2@0.0625_0.25_2_0.9_2000000_10_64_4e-05_0_False_fixedStep@tuning/checkpoints/log.txt`

### Multi GPU Training



![Multi GPU Training](./resources/multiGpuTraining.png)

The framework supports multiple GPU training for a given parameter set. For this, you need to set:
+ set --gres=gpu:8 to desire # GPUs (up to #GPUs/Node)
+ set --cpus=per-task=32 to 4x #GPUs (i.e. 32 for a 8-GPU training)
+ set #workers (WR) in `run_tune.sh` to  8x #GPUs (e.g. 64 for a 8-GPU training)
+ scale the batch_size in the configs with the GPU increase (scale linearly, BS=32 for a single GPU ... and  BS=512 for a 8-GPU setup)
+ You will need to play with other parameters, most omportantlly Learning Rate after changing #GPUs. For a good starting point, if for example LR=0.2 works well on 1 GPU, for a 8-GPU training, it's recommended to schedule the learning rate such that it has a warmup of increases from LR=0.2 to LR=8x0.2 for 5 epochs, and then use a schedule similar to the 1-GPU case.


Note that these numbers are only recommended base on my own experience!

### Helper functions

 + summary_all_models.sh
 + tuning_best.sh: reports best accuracy for group of training
 + tuning_worst.sh

NOTE: put this in your ~/.bashrc file: `alias squeueMe='nvidia-smi2;squeue -u your-username -o "%.18i %120j %20S %10L %.10M %.6D %.2t"'

### Things to set

in `model/parameters.json` ****:
 - add the tuning parameters in json(dictionary) format for each model and sub-configuration similar to this mobilenetV2 example:

```
    "mobilenetv2_baseline_init":{
        "weight_decay":  [4e-05],
        "batch_size":  [64],
        "lr_value": [0.2, 0.3],
        "num_lr_samples": 2
        "lr_gamma": [0.92, 0.95, 0.97],
        "num_lr_gamma_samples": 3,
        "momentum": [0],
        }
```

in `submit_tune.sh` and `RunBench.slurm`:
 - ws: path for scripts. set ${TRAIN_HOME}
 - rs: path to dump outputs. set ${TRAIN_HOME}

in `RunBench_x.slurm`:
 - SING_IMG='.../custom.simg': which singularity image to use


## Models tested

Result of `bash summary_all_models.sh` (NOTE: missing efficientnet-bx capability):

| Model              | tested | parameter count | | Model             | tested | parameter count |
|--------------------|--------|----------------:|-|-------------------|--------|----------------:|
| alexnet            |        |     61100840    | | resnet18          |   :white_check_mark:    |     11689512    |
| densenet121        |        |      7978856    | | resnet34          |   :white_check_mark:    |     21797672    |
| densenet161        |        |     28681000    | | resnet50          |   :white_check_mark:    |     25557032    |
| densenet169        |        |     14149480    | | resnext101_32x8d  |        |     88791336    |
| densenet201        |        |     20013928    | | resnext50_32x4d   |        |     25028904    |
| mobilenetv2        |   :white_check_mark:    |      3504872    | | squeezenet1_0     |   :white_check_mark:    |      1248424    |
| mobilenet_v2       |   :white_check_mark:    |      3504872    | | squeezenet1_1     |   :white_check_mark:    |      1235496    |
| shufflenet_v2_x0_5 |   :white_check_mark:    |      1366792    | | vgg11             |        |    132863336    |
| shufflenet_v2_x1_0 |   :white_check_mark:    |      2278604    | | vgg11_bn          |        |    132868840    |
| shufflenet_v2_x1_5 |   :white_check_mark:    |      3503624    | | vgg13             |        |    133047848    |
| shufflenet_v2_x2_0 |   :white_check_mark:    |      7393996    | | vgg13_bn          |        |    133053736    |
| inception_v3       |        |     27161264    | | vgg16             |   :white_check_mark:    |    138357544    |
| mnasnet0_5         |        |      2218512    | | vgg16_bn          |        |    138365992    |
| mnasnet0_75        |        |      3170208    | | vgg19             |   :white_check_mark:    |    143667240    |
| mnasnet1_0         |        |      4383312    | | vgg19_bn          |        |    143678248    |
| mnasnet1_3         |        |      6282256    | | wide_resnet101_2  |        |    126886696    |
| resnet101          |        |     44549160    | | wide_resnet50_2   |   :white_check_mark:    |     68883240    |
| resnet152          |        |     60192808    | | googlenet         |        |     13004888    |
| efficientnet-b0    |   :white_check_mark:    |      5288548    | | efficientnet-b2   |   :white_check_mark:    |      9109994    |


## Clusters

Compute Canada Beluga:

  - Run under your-accunt name (#SBATCH --account=your-account)
  - Up to 7 days
  - Min time limit of 1 hour per job
  - Max 1000 running jobs
  - /scratch have a max of #1000K files limitation ( moving IMAGENET to /project)
  - Files under /project should have the `group=your-group`
  - Cluster info: run `partition-stats` command
  - Priority stats: run `sshare -l` command
  - Need to load singularity (done in RunBench) if using the Singularity image (i.e. module load singularity)
  - Need to load correct version of python `module load python/3.7.4` (put it in ~/.bashrc)
  - Transfer large data: https://globus.computecanada.ca (https://docs.computecanada.ca/wiki/Globus)
  - Scheduling info: https://docs.computecanada.ca/wiki/Job_scheduling_policies
  - GPU info: https://docs.computecanada.ca/wiki/Using_GPUs_with_Slurm


## File transfer

You need to transfer files with `your-group` group so you don't exceed the quota on /projcts:

For example moving files from /scratch: `rsync --info=progress2 --chown=your-user:your-group -r path_to/IMAGENET-UNCROPPED destination_path/`

parallel data movement:

```

ls src/IMAGENET-UNCROPPED/train/ | xargs -n1 -P32 -I% rsync --info=progress2 --chown=your-user:your-group -r src/IMAGENET-UNCROPPED/train/% dst/IMAGENET-UNCROPPED/train/
ls src/IMAGENET-UNCROPPED/val/   | xargs -n1 -P32 -I% rsync --info=progress2 --chown=your-user:your-group -r src/MAGENET-UNCROPPED/val/%   dst/IMAGENET-UNCROPPED/val/

```

Run the following to test copy speed:
`bash test_transfer.sh`

The output will be :

```
/project/.../your-user --> /project/.../your-user:
test.tar.gz                                                                                                                                                        100%  145MB 205.7MB/s   00:00
/project/.../your-user --> /home/your-user:
test.tar.gz                                                                                                                                                        100%  145MB 217.3MB/s   00:00
/project/.../your-user --> /scratch/your-user:
test.tar.gz                                                                                                                                                        100%  145MB 220.6MB/s   00:00
/home/your-user --> /project/.../your-user:
test.tar.gz                                                                                                                                                        100%  145MB 219.7MB/s   00:00
/home/your-user --> /home/your-user:
test.tar.gz                                                                                                                                                        100%  145MB 219.7MB/s   00:00
/home/your-user --> /scratch/your-user:
test.tar.gz                                                                                                                                                        100%  145MB 220.0MB/s   00:00
/scratch/your-user --> /project/.../your-user:
test.tar.gz                                                                                                                                                        100%  145MB 216.1MB/s   00:00
/scratch/your-user --> /home/your-user:
test.tar.gz                                                                                                                                                        100%  145MB 217.8MB/s   00:00
/scratch/your-user --> /scratch/your-user:
test.tar.gz                                                                                                                                                        100%  145MB 218.2MB/s   00:00
```

Run the following commands to test drive Wr/Rd speed:

`cd /drive_to_test/user/...`

Writing: 
`sync; dd if=/dev/zero of=tempfile bs=1M count=1024; sync`

output:

```
1024+0 records in
1024+0 records out
1073741824 bytes (1.1 GB, 1.0 GiB) copied, 0.993727 s, 1.1 GB/s
```

Reading: 
`dd if=tempfile of=/dev/null bs=1M count=1024`

```
1024+0 records in
1024+0 records out
1073741824 bytes (1.1 GB, 1.0 GiB) copied, 0.251305 s, 4.3 GB/
```
