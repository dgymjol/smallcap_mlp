# CUDA_VISIBLE_DEVICES=1 python train.py --experiments_dir mlp_train --encoder_name openai/clip-vit-large-patch14 --features_dir l14_features --train_mlp
experiment_name="$1"


for var in 8856 17712 26568 35424 44280 53136 61992 70848 79704 88560
do
  CUDA_VISIBLE_DEVICES=0 python infer.py --encoder_name "openai/clip-vit-large-patch14" --features_path "l14_features" --model_path "${experiment_name}/rag_7M_gpt2" --checkpoint_path checkpoint-${var} --infer_test
done