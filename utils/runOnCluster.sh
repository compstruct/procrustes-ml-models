
## MOBILENET V2
#======================================================
#bash tuneIt.sh mobilenetv2 _baseline baseline


## EFFICIENTNET-B2
#======================================================
#bash tuneIt.sh efficientnet-b2 _5M_1 tuning
#bash tuneIt.sh efficientnet-b2 _5M_2 tuning
#bash tuneIt.sh efficientnet-b2 _2M_1 tuning


## RESNET18
#======================================================
#bash tuneIt.sh resnet18 _baseline_init_noMo baseline
#bash tuneIt.sh resnet18 _baseline baseline
#bash tuneIt.sh resnet18 _2M_1 tuning
#bash tuneIt.sh resnet18 _1M_1 tuning
#bash tuneIt.sh resnet18 _2M_init_noMo tuning
#bash tuneIt.sh resnet18 _4M_init_noMo test_rsync
#bash tuneIt.sh resnet18 _3M_init_noMo test_rsync
#bash tuneIt.sh mobilenetv2 _baseline_init_noMo base_mob

#bash tuneIt.sh mobilenetv2 _2M_init_noMo tune_mob

#bash tuneIt.sh mobilenetv2 _tuning_qe tune_qe
bash tuneIt.sh resnet18 _tuning_qe tune_qe
