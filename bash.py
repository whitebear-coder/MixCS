'''
CUDA_VISIBLE_DEVICES=0,2,3 python run.py --root_data_file="/home/linzexu/meta_code/Meta-DMoE-main/CSN" --train_data_file="train.jsonl" --eval_data_file "valid.jsonl" --test_data_file="test.jsonl" --output_dir=./saved_models --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base --do_train --do_eval --do_test --num_train_epochs 10 \
    --block_size 256 --train_batch_size 8 --eval_batch_size 16 --learning_rate 2e-5 --max_grad_norm 1.0 --num_labels 6 --seed 123456  2>&1 | tee train.log
'''
'''
--train_data_file "train.jsonl" --eval_data_file "valid.jsonl" --test_data_file "test.jsonl" --codebase_file "codebase.jsonl" --datasets "javascript" "java" "go" "python" "php" "ruby" --datasets_len 99999999 --huggingface "/home/linzexu/huggingface/unixcode" --train_batch_size 128 --expert_epoch 5 --test_domain_dataset "ruby" "java" "go" "php" "python" "javascript" --train_student_epoch 5 --epoch 5 --aggregator_pretrain_epoch 10
'''

'''
/home/linzexu/meta_code/Meta-DMoE-main/CSN
train.jsonl
valid.jsonl
test.jsonl
'''