wget https://www.dropbox.com/s/2eo6lrnxrj5qg1x/model_vae.ckpt.data-00000-of-00001 -O model/model_vae.ckpt.data-00000-of-00001
wget https://www.dropbox.com/s/in7qxswzbx4tlp7/model_dcgan.ckpt.data-00000-of-00001 -O model/model_dcgan.ckpt.data-00000-of-00001
wget https://www.dropbox.com/s/o281ye37numah9d/model_acgan.ckpt.data-00000-of-00001 -O model/model_acgan.ckpt.data-00000-of-00001

python3 vae.py reconstruct -d=$1 -o=$2
python3 vae.py generate -d=$1 -o=$2

python3 dcgan.py generate -d=$1 -o=$2

python3 acgan.py generate -d=$1 -o=$2

python3 plot_learning_curve.py -o=$2

