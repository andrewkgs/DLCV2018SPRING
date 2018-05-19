wget https://www.dropbox.com/s/2eo6lrnxrj5qg1x/model_vae.ckpt.data-00000-of-00001 -O vae_model/model_vae.ckpt.data-00000-of-00001
python3 vae.py reconstruct -d=$1 -o=$2
python3 vae.py generate -d=$1 -o=$2
