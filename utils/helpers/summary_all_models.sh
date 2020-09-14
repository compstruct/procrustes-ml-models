
ws='/home/aming/MICRO/mobilenetv2/imagenet_pytorch_training'

DIR=$ws/models
model_file=$DIR/models.list

echo 'Model, parameter' > $DIR/model_parameters.csv
while IFS= read -r MODEL
do
    echo "$MODEL $CONFIG $BENCH $FOLDER"
    python summary_model.py -a $MODEL > $DIR/$MODEL.arch
    echo -e "$MODEL,\t $(tail -1 $DIR/$MODEL.arch)" >> $DIR/model_parameters.csv
done < $model_file

