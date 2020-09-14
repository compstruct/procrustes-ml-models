#----------------------------------------------------------

ws='/home/aming/MICRO/mobilenetv2/imagenet_pytorch_training'

LR=$1
LD=$2
LS=$3 #2
ID=$4
TS=$5
EP=$6
BS=$7 #64
WD=$8 #'4e-5'
MO=$9 # momentum
DB=${10} #True/False
ST=${11} #fixedStep / step
QI=${12}
QS=${13}
QF=${14}
MD=${15} # 'mobilenetv2'
IN=${16} # '/datasets/IMAGENET-UNCROPPED' or '/scratch/aming/IMAGENET-UNCROPPED' 

CP=./checkpoints # checkpointing folder
IS=224 # input size in pixels
DA=pytorch #dali-gpu # dataloader backend
WR=8 #64 #8 number of workers
CR=checkpoints/checkpoint.pth.tar # path of checkpoints
QE=True # False
echo "Tuning for LR=$LR LD=$LD LS=$LS ID=$ID WD=$WD TS=$TS EP=$EP BS=$BS MO=$MO DB=$DB ST=$ST QI=$QI QS=$QS QF=$QF MD=$MD IN=$IN"
echo "My home path is: $(pwd)" 
#----------------------------------------------------------

python $ws/imagenet_dropback.py \
    -a $MD \
    -d $IN \
    --epochs $EP \
    --lr-decay $ST \
    --step $LS \
    --gamma=$LD \
    --lr $LR \
    --init-decay $ID \
    --wd $WD \
    -c $CP \
    --input-size $IS \
    --batch-size $BS \
    --track-weights $TS \
    --momentum $MO \
    --drop-back $DB \
    -j $WR \
    --data-backend $DA \
    --q-init $QI \
    --q-step $QS \
    --q-sf $QF \
    --use_qe $QE \
    --resume $CR

#----------------------------------------------------------

#    --early-term True
