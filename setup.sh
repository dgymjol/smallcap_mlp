

# https://github.com/dgymjol/smallcap_mlp.git

# sudo apt-get install libgl1-mesa-glx libglib2.0-0 libgl1

# git config --global user.email dgymjol@yonsei.ac.kr
# git config --global user.name dgymjol


# 1. 
conda create -n smallcap
conda activate smallcap

# 2. 
# for 1080
# conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
# for 3090
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch


# 3.
pip install -r requirements.txt


# 4.(( https://drive.google.com/u/0/uc?id=1ZP5I-xbjaNU7cU48C_ctHd95SaA0jBHe&export=download ))
mkdir datastore
cd datastore
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ZP5I-xbjaNU7cU48C_ctHd95SaA0jBHe' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ZP5I-xbjaNU7cU48C_ctHd95SaA0jBHe" -O coco_index && rm -rf /tmp/cookies.txt
cd cd../


# 5. (( https://drive.google.com/file/d/1BT0Qc6g40fvtnJ_yY0aipfCuCMgu5qaR/view
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1BT0Qc6g40fvtnJ_yY0aipfCuCMgu5qaR' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1BT0Qc6g40fvtnJ_yY0aipfCuCMgu5qaR" -O coco_index_captions.json && rm -rf /tmp/cookies.txt


# 6. file dataset_coco.json from here and place it in data/.


# 7. download coco dataset 

curl -O http://images.cocodataset.org/zips/val2017.zip
curl -O http://images.cocodataset.org/zips/train2017.zip 
curl -O http://images.cocodataset.org/zips/test2017.zip 

unzip val2017.zip
unzip train2017.zip
unzip test2017.zip

mv val2017/*.jpg train2017
mv test2017/*.jpg train2017

rm -rf val2017
rm -rf test2017

mv train2017 data/images

# 8.
pip install git+https://github.com/openai/CLIP.git

# 9.
mkdir features
python src/extract_features.py

# 10.
python src/retrieve_caps.py


# 11. Model training
python train.py