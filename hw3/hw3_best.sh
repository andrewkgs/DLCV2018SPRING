mkdir -p model
wget https://www.dropbox.com/s/nzdxsv5bmzp7knc/model_VGG16FCN16s.h5
mv model_VGG16FCN16s.h5 ./model/

python3 semantic_segmentation.py test VGG16-FCN16s --test_data_dir=$1 --predict_dir=$2 --model_file='./model/model_VGG16FCN16s.h5'
