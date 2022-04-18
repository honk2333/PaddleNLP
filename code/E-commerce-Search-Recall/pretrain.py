# 对模型进行MLM预训练
from transformers import AutoModelForMaskedLM,AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import LineByLineTextDataset
import os
import math

# model_dir = "../model"
# model_name = "roberta-large"
train_file = "../data/pretrain_data.txt"
eval_file = "../data/pretrain_evaldata.txt"
max_seq_length = 64
out_model_path = os.path.join('SimCSE/','output')
print(out_model_path)
train_epoches = 10
batch_size = 96

# 这里不是从零训练，而是在原有预训练的基础上增加数据进行预训练，因此不会从 config 导入模型
tokenizer = AutoTokenizer.from_pretrained(
            '/home/data_ti6_c/wanghk/bert_model/chinese-roberta-wwm-ext')

# 继续预训练
model = AutoModelForMaskedLM.from_pretrained(
    '/home/wanghk/PaddleNLP/code/E-commerce-Search-Recall/SimCSE/output/checkpoint')

# model = AutoModelForMaskedLM.from_pretrained(
#     '/home/data_ti6_c/wanghk/bert_model/chinese-roberta-wwm-ext')

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=train_file,
    block_size=128,
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

eval_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=eval_file,
    block_size=128,
)

training_args = TrainingArguments(
        output_dir=out_model_path,
        overwrite_output_dir=True,
        num_train_epochs=train_epoches,
        per_device_train_batch_size=batch_size,
        save_steps=2000,
        save_total_limit=4,
        prediction_loss_only=True,
    )


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

trainer.train()

trainer.save_model(out_model_path)
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")