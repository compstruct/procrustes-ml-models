
cd pdfs

mkdir resnet
mkdir mobilenet
mkdir efficientnet

mv *_resnet* resnet
mv *_mobile* mobilenet
mv *_eff* efficientnet

for folder in resnet mobilenet efficientnet
do
cd $folder

mkdir baseline
mkdir tuning
mv *baseline* baseline
mv *tuning* tuning

cd ..
done

