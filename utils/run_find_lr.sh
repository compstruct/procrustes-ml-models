#----------------------------------------------------------

ws='/home/aming/MICRO/mobilenetv2/imagenet_pytorch_training'

LR=0.1 #$1
LD='1' #$2
LS=1 #$3 #2
ID=0.9 #$4
TS=2000000 #$5
EP=1 #$6
BS=32 #$7 #64
WD=0 #$8 #'4e-5'
MO=0 #$9 # momentum
DB='True' #/False
ST='fixedStep' #${11} #fixedStep / step
MD='resnet18' #${12} # 'mobilenetv2'
IN='/datasets/IMAGENET-UNCROPPED' #or '/scratch/aming/IMAGENET-UNCROPPED' 

IS=224 # input size in pixels
DA=dali-gpu # dataloader backend
WR=4 # number of workers
CR=checkpoints/checkpoint.pth.tar # path of checkpoints

echo "Tuning for LR=$LR LD=$LD LS=$LS ID=$ID WD=$WD TS=$TS EP=$EP BS=$BS MO=$MO DB=$DB ST=$ST MD=$MD IN=$IN"
echo "My home path is: $(pwd)" 
#----------------------------------------------------------

python $ws/find_lr.py \
    -a $MD \
    -d $IN \
    --lr-decay $ST \
    --lr $LR \
    --init-decay $ID \
    --wd $WD \
    --input-size $IS \
    --batch-size $BS \
    --track-weights $TS \
    --momentum $MO \
    --drop-back $DB \
    -j $WR \
    --data-backend $DA

#----------------------------------------------------------

scp find_lr.png rigel:~
