# run this on jarvis 2 only

bash collect_all_logs.sh
ssh jarvis1 -t "cd ~/MICRO/mobilenetv2/imagenet_pytorch_training; bash collect_all_logs.sh"
ssh jarvis3 -t "cd ~/MICRO/mobilenetv2/imagenet_pytorch_training; bash collect_all_logs.sh"

scp -r jarvis1:~/MICRO/mobilenetv2/imagenet_pytorch_training/logs .
scp -r jarvis3:~/MICRO/mobilenetv2/imagenet_pytorch_training/logs .

for f in logs/*
do
    echo $f
    python3 plot_log.py $f

done

cp logs/*.pdf pdfs/
bash organize_pdfs.sh

scp -r pdfs rigel:~
