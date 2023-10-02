experiment_name="$1"

######## Model training (MUST BE transformers 4.21.1)

CUDA_VISIBLE_DEVICES=0 python train.py --experiments_dir "${experiment_name}"

mkdir "${experiment_name}/results"

######## validation (val set) (If you specify --infer_test inference uses test data, else val data is used.)

CUDA_VISIBLE_DEVICES=0 python infer.py --model_path "${experiment_name}/rag_7M_gpt2" --checkpoint_path checkpoint-8856
CUDA_VISIBLE_DEVICES=0 python infer.py --model_path "${experiment_name}/rag_7M_gpt2" --checkpoint_path checkpoint-17712
CUDA_VISIBLE_DEVICES=0 python infer.py --model_path "${experiment_name}/rag_7M_gpt2" --checkpoint_path checkpoint-26568
CUDA_VISIBLE_DEVICES=0 python infer.py --model_path "${experiment_name}/rag_7M_gpt2" --checkpoint_path checkpoint-35424
CUDA_VISIBLE_DEVICES=0 python infer.py --model_path "${experiment_name}/rag_7M_gpt2" --checkpoint_path checkpoint-44280
CUDA_VISIBLE_DEVICES=0 python infer.py --model_path "${experiment_name}/rag_7M_gpt2" --checkpoint_path checkpoint-53136
CUDA_VISIBLE_DEVICES=0 python infer.py --model_path "${experiment_name}/rag_7M_gpt2" --checkpoint_path checkpoint-61992
CUDA_VISIBLE_DEVICES=0 python infer.py --model_path "${experiment_name}/rag_7M_gpt2" --checkpoint_path checkpoint-70848
CUDA_VISIBLE_DEVICES=0 python infer.py --model_path "${experiment_name}/rag_7M_gpt2" --checkpoint_path checkpoint-79704
CUDA_VISIBLE_DEVICES=0 python infer.py --model_path "${experiment_name}/rag_7M_gpt2" --checkpoint_path checkpoint-88560

CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json "${experiment_name}/rag_7M_gpt2/checkpoint-8856/val_preds.json" > "${experiment_name}/results/val_8856.txt"
CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json "${experiment_name}/rag_7M_gpt2/checkpoint-17712/val_preds.json" > "${experiment_name}/results/val_17712.txt"
CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json "${experiment_name}/rag_7M_gpt2/checkpoint-26568/val_preds.json" > "${experiment_name}/results/val_26568.txt"
CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json "${experiment_name}/rag_7M_gpt2/checkpoint-35424/val_preds.json" > "${experiment_name}/results/val_35424.txt"
CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json "${experiment_name}/rag_7M_gpt2/checkpoint-44280/val_preds.json" > "${experiment_name}/results/val_44280.txt"
CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json "${experiment_name}/rag_7M_gpt2/checkpoint-53136/val_preds.json" > "${experiment_name}/results/val_53136.txt"
CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json "${experiment_name}/rag_7M_gpt2/checkpoint-61992/val_preds.json" > "${experiment_name}/results/val_61992.txt"
CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json "${experiment_name}/rag_7M_gpt2/checkpoint-70848/val_preds.json" > "${experiment_name}/results/val_70848.txt"
CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json "${experiment_name}/rag_7M_gpt2/checkpoint-79704/val_preds.json" > "${experiment_name}/results/val_79704.txt"
CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json "${experiment_name}/rag_7M_gpt2/checkpoint-88560/val_preds.json" > "${experiment_name}/results/val_88560.txt"



######## Test (test for best val set)

CUDA_VISIBLE_DEVICES=0 python infer.py --model_path "${experiment_name}/rag_7M_gpt2" --checkpoint_path checkpoint-8856 --infer_test
CUDA_VISIBLE_DEVICES=0 python infer.py --model_path "${experiment_name}/rag_7M_gpt2" --checkpoint_path checkpoint-17712 --infer_test
CUDA_VISIBLE_DEVICES=0 python infer.py --model_path "${experiment_name}/rag_7M_gpt2" --checkpoint_path checkpoint-26568 --infer_test
CUDA_VISIBLE_DEVICES=0 python infer.py --model_path "${experiment_name}/rag_7M_gpt2" --checkpoint_path checkpoint-35424 --infer_test
CUDA_VISIBLE_DEVICES=0 python infer.py --model_path "${experiment_name}/rag_7M_gpt2" --checkpoint_path checkpoint-44280 --infer_test
CUDA_VISIBLE_DEVICES=0 python infer.py --model_path "${experiment_name}/rag_7M_gpt2" --checkpoint_path checkpoint-53136 --infer_test
CUDA_VISIBLE_DEVICES=0 python infer.py --model_path "${experiment_name}/rag_7M_gpt2" --checkpoint_path checkpoint-61992 --infer_test
CUDA_VISIBLE_DEVICES=0 python infer.py --model_path "${experiment_name}/rag_7M_gpt2" --checkpoint_path checkpoint-70848 --infer_test
CUDA_VISIBLE_DEVICES=0 python infer.py --model_path "${experiment_name}/rag_7M_gpt2" --checkpoint_path checkpoint-79704 --infer_test
CUDA_VISIBLE_DEVICES=0 python infer.py --model_path "${experiment_name}/rag_7M_gpt2" --checkpoint_path checkpoint-88560 --infer_test

CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json "${experiment_name}/rag_7M_gpt2/checkpoint-8856/test_preds.json" > "${experiment_name}/results/test_8856.txt"
CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json "${experiment_name}/rag_7M_gpt2/checkpoint-17712/test_preds.json" > "${experiment_name}/results/test_17712.txt"
CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json "${experiment_name}/rag_7M_gpt2/checkpoint-26568/test_preds.json" > "${experiment_name}/results/test_26568.txt"
CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json "${experiment_name}/rag_7M_gpt2/checkpoint-35424/test_preds.json" > "${experiment_name}/results/test_35424.txt"
CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json "${experiment_name}/rag_7M_gpt2/checkpoint-44280/test_preds.json" > "${experiment_name}/results/test_44280.txt"
CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json "${experiment_name}/rag_7M_gpt2/checkpoint-53136/test_preds.json" > "${experiment_name}/results/test_53136.txt"
CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json "${experiment_name}/rag_7M_gpt2/checkpoint-61992/test_preds.json" > "${experiment_name}/results/test_61992.txt"
CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json "${experiment_name}/rag_7M_gpt2/checkpoint-70848/test_preds.json" > "${experiment_name}/results/test_70848.txt"
CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json "${experiment_name}/rag_7M_gpt2/checkpoint-79704/test_preds.json" > "${experiment_name}/results/test_79704.txt"
CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json "${experiment_name}/rag_7M_gpt2/checkpoint-88560/test_preds.json" > "${experiment_name}/results/test_88560.txt"
