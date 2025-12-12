export PYTHON=../

python ../infer.py --dataset real --nerf_L 10 --bert_input_dim 63 --masking --ordering --num_layers 6 --model_path ../weights/all_class/df_epoch_best_multicls.pth --exp_name run_multi_class_pretrain --mask_per 0
