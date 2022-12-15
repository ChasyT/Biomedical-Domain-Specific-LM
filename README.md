# Biomedical Domain-Specific LM
This code is a final course project of NLPDL, 2022 Fall. 

Research shows that the most common way to improve the performance of **domain-specific** NLP task fine-tuning is to use the language model of corporate-specific **post-training**. By collecting a large amount of unlabeled data in this field and continuing MLM (mask language model) pre-training on Roberta and other pre-training models, the fine-tuning effect of downstream tasks can be significantly improved.

I use *roberta-base* as the basic model, and use *BIOMRC*, a large-scale cloze-style biomedical MRC dataset, as the post-training corpus. And the post-training model shows great performance on fine-tuning task: Chemical-protein Interaction Prediction(CHEMPROT) and QA(BioASQ).

Based on that, I have tried three methods to improve the performance. Although the results are not significant, I will also show how to reproduce the methods here.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

### Post-training

Use ***roberta-base*** as the basic model. Use *biomrc/biomrc_small_A* (https://huggingface.co/datasets/biomrc) as the post-training corpus.

```bash
cd train

python run_mlm.py \
	--model_name_or_path roberta-base \
	--dataset_name biomrc \
	--dataset_config_name biomrc_small_A \
	--do_train \
	--output_dir ./post_train
```

The checkpoints will be saved in `./train/post_train`.

### Fine-tuning

Take dataset **BioASQ** for example:

```bash
python train.py \
	--model_name_or_path post_train \
	--vocab_path post_train \
	--dataset_name bioasq \
	--seed 2022
```

If you want to train dataset **CHEMPROT**, please replace `bioasq` with `chemprot`. Here, the 13 labels of CHEMPROT are grouped into five categories.

If you want to train few-shot dataset, you can try `bioasq_fs` and `chemprot_fs`.

The results will be saved at file `results`.

## Other Methods

### Post-training with Adapter

On the basis of **post-training**, add `--add_adapter True`.

### Fine-tuning with Adapted Vocab

Take dataset **BioASQ** for example:

```
cd avocado

python avocado.py \
	--dataset bioasq \
	--root ../../dataloader \
	--vocab_size 10000 \
	--use_fragment \
	--encoder_class ../post_train
```

This will generate file `post_train10000-merged-vocabulary_optimized_frt` at file `dataloader`. Please move it to `dataloader/bioasq` for further training.

Now, we can use it as `--vocab_path` at fine-tuning.

### Post-training with Larger batch_size

The larger the batch_size used for post-training, the better the results. The setting of max_length has a great impact on the memory usage, which can be properly selected to facilitate the adjustment of batch size.

You can try to increase batch_size by `--per_device_train_batch_size`, and meanwhile decrease max_length by `--max_seq_length`. 

## Acknowledgement

This code refers to https://github.com/huggingface/transformers.

The code related to AVocaDo is modified from https://github.com/Jimin9401/avocado.

## Reference

[1]. Gu Y, Tinn R, Cheng H, et al. Domain-specific language model pretraining for biomedical natural language processing[J]. ACM Transactions on Computing for Healthcare (HEALTH), 2021, 3(1): 1-23.

[2]. Hong J, Kim T, Lim H, et al. AVocaDo: Strategy for adapting vocabulary to downstream domain[J]. arXiv preprint arXiv:2110.13434, 2021.
