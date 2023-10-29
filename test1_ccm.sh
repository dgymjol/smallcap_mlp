experiment_name="$1"

# for ccm pretrained weight (finetuning)


lr=5e-4
CUDA_VISIBLE_DEVICES=1 python train.py --experiments_dir ${experiment_name}_finetuning_${lr} --encoder_name openai/clip-vit-large-patch14 --features_dir l14_features --train_mlp --mlp_path pic2word.pt --lr $lr
mkdir "${experiment_name}_finetuning_${lr}/results"
for var in 8856 17712 26568 35424 44280 53136 61992 70848 79704 88560
do
  CUDA_VISIBLE_DEVICES=1 python infer.py --encoder_name "openai/clip-vit-large-patch14" --features_path "l14_features/val.hdf5" --model_path "${experiment_name}_finetuning_${lr}/rag_7M_gpt2" --checkpoint_path checkpoint-${var}
  CUDA_VISIBLE_DEVICES=1 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json "${experiment_name}_finetuning_${lr}/rag_7M_gpt2/checkpoint-${var}/val_preds.json" > "${experiment_name}_finetuning_${lr}/results/val_${var}.txt"

  CUDA_VISIBLE_DEVICES=1 python infer.py --encoder_name "openai/clip-vit-large-patch14" --model_path "${experiment_name}_finetuning_${lr}/rag_7M_gpt2" --checkpoint_path checkpoint-${var} --infer_test
  CUDA_VISIBLE_DEVICES=1 python coco-caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json "${experiment_name}_finetuning_${lr}/rag_7M_gpt2/checkpoint-${var}/test_preds.json" > "${experiment_name}_finetuning_${lr}/results/test_${var}.txt"
done

lr=1e-4
CUDA_VISIBLE_DEVICES=1 python train.py --experiments_dir ${experiment_name}_finetuning_${lr} --encoder_name openai/clip-vit-large-patch14 --features_dir l14_features --train_mlp --mlp_path pic2word.pt --lr $lr
mkdir "${experiment_name}_finetuning_${lr}/results"
for var in 8856 17712 26568 35424 44280 53136 61992 70848 79704 88560
do
  CUDA_VISIBLE_DEVICES=1 python infer.py --encoder_name "openai/clip-vit-large-patch14" --features_path "l14_features/val.hdf5" --model_path "${experiment_name}_finetuning_${lr}/rag_7M_gpt2" --checkpoint_path checkpoint-${var}
  CUDA_VISIBLE_DEVICES=1 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json "${experiment_name}_finetuning_${lr}/rag_7M_gpt2/checkpoint-${var}/val_preds.json" > "${experiment_name}_finetuning_${lr}/results/val_${var}.txt"

  CUDA_VISIBLE_DEVICES=1 python infer.py --encoder_name "openai/clip-vit-large-patch14" --model_path "${experiment_name}_finetuning_${lr}/rag_7M_gpt2" --checkpoint_path checkpoint-${var} --infer_test
  CUDA_VISIBLE_DEVICES=1 python coco-caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json "${experiment_name}_finetuning_${lr}/rag_7M_gpt2/checkpoint-${var}/test_preds.json" > "${experiment_name}_finetuning_${lr}/results/test_${var}.txt"
done

lr=5e-5
CUDA_VISIBLE_DEVICES=1 python train.py --experiments_dir ${experiment_name}_finetuning_${lr} --encoder_name openai/clip-vit-large-patch14 --features_dir l14_features --train_mlp --mlp_path pic2word.pt --lr $lr
mkdir "${experiment_name}_finetuning_${lr}/results"
for var in 8856 17712 26568 35424 44280 53136 61992 70848 79704 88560
do
  CUDA_VISIBLE_DEVICES=1 python infer.py --encoder_name "openai/clip-vit-large-patch14" --features_path "l14_features/val.hdf5" --model_path "${experiment_name}_finetuning_${lr}/rag_7M_gpt2" --checkpoint_path checkpoint-${var}
  CUDA_VISIBLE_DEVICES=1 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json "${experiment_name}_finetuning_${lr}/rag_7M_gpt2/checkpoint-${var}/val_preds.json" > "${experiment_name}_finetuning_${lr}/results/val_${var}.txt"

  CUDA_VISIBLE_DEVICES=1 python infer.py --encoder_name "openai/clip-vit-large-patch14" --model_path "${experiment_name}_finetuning_${lr}/rag_7M_gpt2" --checkpoint_path checkpoint-${var} --infer_test
  CUDA_VISIBLE_DEVICES=1 python coco-caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json "${experiment_name}_finetuning_${lr}/rag_7M_gpt2/checkpoint-${var}/test_preds.json" > "${experiment_name}_finetuning_${lr}/results/test_${var}.txt"
done

lr=1e-5
CUDA_VISIBLE_DEVICES=1 python train.py --experiments_dir ${experiment_name}_finetuning_${lr} --encoder_name openai/clip-vit-large-patch14 --features_dir l14_features --train_mlp --mlp_path pic2word.pt --lr $lr
mkdir "${experiment_name}_finetuning_${lr}/results"
for var in 8856 17712 26568 35424 44280 53136 61992 70848 79704 88560
do
  CUDA_VISIBLE_DEVICES=1 python infer.py --encoder_name "openai/clip-vit-large-patch14" --features_path "l14_features/val.hdf5" --model_path "${experiment_name}_finetuning_${lr}/rag_7M_gpt2" --checkpoint_path checkpoint-${var}
  CUDA_VISIBLE_DEVICES=1 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json "${experiment_name}_finetuning_${lr}/rag_7M_gpt2/checkpoint-${var}/val_preds.json" > "${experiment_name}_finetuning_${lr}/results/val_${var}.txt"

  CUDA_VISIBLE_DEVICES=1 python infer.py --encoder_name "openai/clip-vit-large-patch14" --model_path "${experiment_name}_finetuning_${lr}/rag_7M_gpt2" --checkpoint_path checkpoint-${var} --infer_test
  CUDA_VISIBLE_DEVICES=1 python coco-caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json "${experiment_name}_finetuning_${lr}/rag_7M_gpt2/checkpoint-${var}/test_preds.json" > "${experiment_name}_finetuning_${lr}/results/test_${var}.txt"
done


# for ccm pretrained weight (freeze)


lr=5e-4
CUDA_VISIBLE_DEVICES=1 python train.py --experiments_dir ${experiment_name}_freeze_${lr} --encoder_name openai/clip-vit-large-patch14 --features_dir l14_features --mlp_path pic2word.pt --lr $lr
mkdir "${experiment_name}_freeze_${lr}/results"
for var in 8856 17712 26568 35424 44280 53136 61992 70848 79704 88560
do
  CUDA_VISIBLE_DEVICES=1 python infer.py --encoder_name "openai/clip-vit-large-patch14" --features_path "l14_features/val.hdf5" --model_path "${experiment_name}_freeze_${lr}/rag_7M_gpt2" --checkpoint_path checkpoint-${var}
  CUDA_VISIBLE_DEVICES=1 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json "${experiment_name}_freeze_${lr}/rag_7M_gpt2/checkpoint-${var}/val_preds.json" > "${experiment_name}_freeze_${lr}/results/val_${var}.txt"

  CUDA_VISIBLE_DEVICES=1 python infer.py --encoder_name "openai/clip-vit-large-patch14" --model_path "${experiment_name}_freeze_${lr}/rag_7M_gpt2" --checkpoint_path checkpoint-${var} --infer_test
  CUDA_VISIBLE_DEVICES=1 python coco-caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json "${experiment_name}_freeze_${lr}/rag_7M_gpt2/checkpoint-${var}/test_preds.json" > "${experiment_name}_freeze_${lr}/results/test_${var}.txt"
done

lr=1e-4
CUDA_VISIBLE_DEVICES=1 python train.py --experiments_dir ${experiment_name}_freeze_${lr} --encoder_name openai/clip-vit-large-patch14 --features_dir l14_features --mlp_path pic2word.pt --lr $lr
mkdir "${experiment_name}_freeze_${lr}/results"
for var in 8856 17712 26568 35424 44280 53136 61992 70848 79704 88560
do
  CUDA_VISIBLE_DEVICES=1 python infer.py --encoder_name "openai/clip-vit-large-patch14" --features_path "l14_features/val.hdf5" --model_path "${experiment_name}_freeze_${lr}/rag_7M_gpt2" --checkpoint_path checkpoint-${var}
  CUDA_VISIBLE_DEVICES=1 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json "${experiment_name}_freeze_${lr}/rag_7M_gpt2/checkpoint-${var}/val_preds.json" > "${experiment_name}_freeze_${lr}/results/val_${var}.txt"

  CUDA_VISIBLE_DEVICES=1 python infer.py --encoder_name "openai/clip-vit-large-patch14" --model_path "${experiment_name}_freeze_${lr}/rag_7M_gpt2" --checkpoint_path checkpoint-${var} --infer_test
  CUDA_VISIBLE_DEVICES=1 python coco-caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json "${experiment_name}_freeze_${lr}/rag_7M_gpt2/checkpoint-${var}/test_preds.json" > "${experiment_name}_freeze_${lr}/results/test_${var}.txt"
done

lr=5e-5
CUDA_VISIBLE_DEVICES=1 python train.py --experiments_dir ${experiment_name}_freeze_${lr} --encoder_name openai/clip-vit-large-patch14 --features_dir l14_features --mlp_path pic2word.pt --lr $lr
mkdir "${experiment_name}_freeze_${lr}/results"
for var in 8856 17712 26568 35424 44280 53136 61992 70848 79704 88560
do
  CUDA_VISIBLE_DEVICES=1 python infer.py --encoder_name "openai/clip-vit-large-patch14" --features_path "l14_features/val.hdf5" --model_path "${experiment_name}_freeze_${lr}/rag_7M_gpt2" --checkpoint_path checkpoint-${var}
  CUDA_VISIBLE_DEVICES=1 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json "${experiment_name}_freeze_${lr}/rag_7M_gpt2/checkpoint-${var}/val_preds.json" > "${experiment_name}_freeze_${lr}/results/val_${var}.txt"

  CUDA_VISIBLE_DEVICES=1 python infer.py --encoder_name "openai/clip-vit-large-patch14" --model_path "${experiment_name}_freeze_${lr}/rag_7M_gpt2" --checkpoint_path checkpoint-${var} --infer_test
  CUDA_VISIBLE_DEVICES=1 python coco-caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json "${experiment_name}_freeze_${lr}/rag_7M_gpt2/checkpoint-${var}/test_preds.json" > "${experiment_name}_freeze_${lr}/results/test_${var}.txt"
done

lr=1e-5
CUDA_VISIBLE_DEVICES=1 python train.py --experiments_dir ${experiment_name}_freeze_${lr} --encoder_name openai/clip-vit-large-patch14 --features_dir l14_features --mlp_path pic2word.pt --lr $lr
mkdir "${experiment_name}_freeze_${lr}/results"
for var in 8856 17712 26568 35424 44280 53136 61992 70848 79704 88560
do
  CUDA_VISIBLE_DEVICES=1 python infer.py --encoder_name "openai/clip-vit-large-patch14" --features_path "l14_features/val.hdf5" --model_path "${experiment_name}_freeze_${lr}/rag_7M_gpt2" --checkpoint_path checkpoint-${var}
  CUDA_VISIBLE_DEVICES=1 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json "${experiment_name}_freeze_${lr}/rag_7M_gpt2/checkpoint-${var}/val_preds.json" > "${experiment_name}_freeze_${lr}/results/val_${var}.txt"

  CUDA_VISIBLE_DEVICES=1 python infer.py --encoder_name "openai/clip-vit-large-patch14" --model_path "${experiment_name}_freeze_${lr}/rag_7M_gpt2" --checkpoint_path checkpoint-${var} --infer_test
  CUDA_VISIBLE_DEVICES=1 python coco-caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json "${experiment_name}_freeze_${lr}/rag_7M_gpt2/checkpoint-${var}/test_preds.json" > "${experiment_name}_freeze_${lr}/results/test_${var}.txt"
done
