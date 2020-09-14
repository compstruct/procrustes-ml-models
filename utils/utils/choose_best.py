import pandas as pd
import os

def best_accuracy(current_epoch=0, run_name='ImageNet_mobilenetv2', folder='baseline'):
    best_file='/home/aming/MICRO/mobilenetv2/imagenet_pytorch_training/'+folder+'/tmps/'+run_name+'/tuning_best.csv'

    if not os.path.exists(best_file):
       print('tuning_best.csv for this entry is not generated!')
       return 0

    cereal_df=''
    valid_acc=0
    try:
        cereal_df = pd.read_csv(best_file)
        valid_acc=cereal_df['Valid Acc.']
    except:
        print('tuning_best.csv seems empty!')
        return 0

    if len(valid_acc) < 1:
        print('tuning_best.csv seems have no entries!')
        return 0

    if current_epoch < len(valid_acc):
        return valid_acc[current_epoch]
    else: # report the last accuracy reported
        return valid_acc[len(valid_acc)-1]
#print(best_accuracy(9) )
