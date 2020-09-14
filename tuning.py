from random import random, sample , seed
import sys
import json

## ----------------------------------------------------------------------
# init
seed(1) #to make sure we can reproduce the same parameters

# populate this per model you want to tune!

## ----------------------------------------------------------------------

model_name='mobilenetv2'
model_id='_2M_1'

if len(sys.argv) < 3:
    raise 'Missing Model Name!!'
else:
    model_name= str(sys.argv[1])
    model_id = str(sys.argv[2])

## ----------------------------------------------------------------------

model_name_id=model_name + model_id
hyper_setups = json.load(open('models/parameters.json','r'))
hyper_setup=hyper_setups['MASTER']
requested=hyper_setups[model_name_id]
hyper_setup.update(requested)



## Default values ##
weight_decay = hyper_setup['weight_decay']          # list of int
init_decay = hyper_setup['init_decay']              # list of int
batch_size = hyper_setup['batch_size']              # list of int
epoch = hyper_setup['epoch']                        # single int
momentum = hyper_setup['momentum']          # list of int
drop_back = hyper_setup['drop_back']          # string
step_type = hyper_setup['step_type']          # string
## Track Size
track_size = hyper_setup['track_size']              # list of int

## learning rates ##
# LR Gamma / Freq
lr_gamma_freq = hyper_setup['lr_gamma_freq']        # single int
lr_gamma_warmup = hyper_setup['lr_gamma_warmup']    # single int
min_gamma= hyper_setup['lr_gamma_freq_min']         # single int
max_gamma = hyper_setup['lr_gamma_freq_max']        # single int
lr_gamma = hyper_setup['lr_gamma']
#lr_gamma = [(min_gamma + i)/100 for i in range(max_gamma-min_gamma)]
lr_gamma_samples = sample(lr_gamma, hyper_setup['num_lr_gamma_samples'])
# LR Value
lr_value = hyper_setup['lr_value']
#lr_value_b= [(50 + i)/100 for i in range(80-50)] #bigger than 0.5
#lr_value_b = sample(lr_value_b, hyper_setup['num_lr_samples'])
#lr_value_s = [2**(-1*(i++2)) for i in range(5)] # smaller than 0.5
#lr_value_s = sample(lr_value_s, hyper_setup['num_lr_samples'])
#lr_value=lr_value_s+lr_value_b
lr_samples = sample(lr_value, hyper_setup['num_lr_samples'])

q_inits = hyper_setup['q_inits']
q_steps = hyper_setup['q_steps']
q_sfs = hyper_setup['q_sfs']


## ----------------------------------------------------------------------
run_samples=[]
print('grid----------------------------------------------------')
ep = epoch
db =drop_back
st= step_type
for mo in momentum:
    for wd in weight_decay:
        for bs in batch_size:
                for Id in init_decay:
                    for ts in track_size:
                        print('track_size= %d' % (ts))
                        for lr_gf in lr_gamma_freq:
                            print('lr_gamma_freq= %d' % (lr_gf))
                            for ds in lr_gamma:
                                if ds in  lr_gamma_samples:
                                    print('lr_gamma:%01.2f*|\t' %(ds)),
                                else:
                                    print('lr_gamma:%01.2f |\t' %(ds)),
                                for ls in  lr_value:
                                    if ds in lr_gamma_samples and ls in  lr_samples:
                                        for qi in  q_inits:
                                            for qs in  q_steps:
                                                for qf in  q_sfs:
                                                    print('%01.4f,%01.4f*|\t' % (ds,ls)),
                                                    run_samples.append({'lr_v':ls,
                                                            'lr_d':ds,
                                                            'lr_gf':lr_gf,
                                                            'id':Id,
                                                            'ts':ts,
                                                            'ep':ep,
                                                            'bs':bs,
                                                            'wd':wd,
                                                            'mo':mo,
                                                            'db':db,
                                                            'st':st,
                                                            'qi':qi,
                                                            'qs':qs,
                                                            'qf':qf
                                                            })
                                    else:
                                        print('%01.4f,%01.4f |\t' % (ds,ls)),
                                print('')
print('----------------------------------------------------')

#print(run_samples)
tuningRun_file="tuning.runs"
run_file = open(tuningRun_file, "w+")

for rs in run_samples:
    s= '_'.join([   str(rs['lr_v']),
                    str(rs['lr_d']),
                    str(rs['lr_gf']),
                    str(rs['id']),
                    str(rs['ts']),
                    str(rs['ep']),
                    str(rs['bs']),
                    str(rs['wd']),
                    str(rs['mo']),
                    str(rs['db']),
                    str(rs['st']),
                    str(rs['qi']),
                    str(rs['qs']),
                    str(rs['qf'])
                ])
    run_file.write(s+'\n')
