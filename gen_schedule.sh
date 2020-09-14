
job_name=$1
GPUs=2
python utils/schedule_generator.py 0.2 0.95 $GPUs 4 150 > schedules/${job_name}.csv
