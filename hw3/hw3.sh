mkdir -p model
wget https://www.dropbox.com/s/od0at7ynrpq8pcj/model_VGG16FCN32s.h5
mv model_VGG16FCN32s.h5 ./model/

python3 semantic_segmentation.py test VGG16-FCN32s --test_data_dir=$1 --predict_dir=$2 --model_file='./model/model_VGG16FCN32s.h5'
