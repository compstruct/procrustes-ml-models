import sys

LR_init =  float(sys.argv[1]) #0.2 --> 1 GPU LR
LR_decay = float(sys.argv[2]) #0.95
lin_scale = int(sys.argv[3]) #2 --> 2x GPU LR= 0.4
warmup_steps = int(sys.argv[4]) #4 in 5 steps
epochs = int(sys.argv[5]) #150

last_LR=LR_init
step= float(lin_scale)**(1/float(warmup_steps))

print("{}\t{}".format(last_LR, step))
for f in range(0,warmup_steps):
    last_LR = last_LR * step
    print("{}\t{}".format(last_LR, step))

step=LR_decay
for f in range(0,epochs-warmup_steps):
    last_LR = last_LR * step
    print("{}\t{}".format(last_LR, step))

