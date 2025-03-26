import os
import torch
import numpy as np
from datetime import datetime
from evaluate import load
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

"""
sft：在下游数据集上微调预训练模型
"""

# 设置存放数据集和模型的根目录
data_path = ".\data\GLUE\MRPC"
model_path = ".\models\\bert-base-uncased"  # 预训练模型存放位置
save_path = ".\output"  # 输出存放位置
version = datetime.now().strftime("%Y%m%d_%H%M")  # 版本控制
checkpoint = f"bert-base-mrpc-{version}"

# 加载数据集
## 如果不是公开数据集而是自己制作的数据集，还要专门添加一个函数或类将数据集处理成标准格式
## 能不能将预处理数据集 + 分词操作放到同一个类中？这样就可以直接获得模型的input_ids了
raw_datasets = load_dataset("parquet",
                            data_files={
                                "train": os.path.join(data_path, "train-00000-of-00001.parquet"),
                                "validation": os.path.join(data_path, "validation-00000-of-00001.parquet"),
                                "test": os.path.join(data_path, "test-00000-of-00001.parquet")
                            })


# 加载预训练分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 使用tokenizeier对数据集进行分词
def tokenize_function(examples):
    # 这里rokenizer里的内容根据任务的不同而不同
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# 动态填充，将每个批次的输入序列填充到一样的长度
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 加载预训练模型
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')


# 根据显存自动计算梯度累积步数
# batch_size = 16 if torch.cuda.is_available() else 4
# max_grad_accum = 2048 // batch_size  # 假设目标总batch_size为2048
# gradient_accumulation_steps = min(max_grad_accum, 8)


# 加载训练参数
# 注：TrainingArguments唯一一个必须传入的参数是保存model的路径
training_args = TrainingArguments(
    output_dir=save_path,
    run_name=checkpoint,
    num_train_epochs=5,  # 训练5个epoch(当数据量小的时候要降低轮数防止过拟合，有early stop吗)
    per_device_train_batch_size=16,  # 每个GPU训练16个batch(确保显存足够)
    per_device_eval_batch_size=32,  # 每个GPU评估32个batch(评估时无需反向传播，可更大)
    gradient_accumulation_steps=4,  # 梯度累积步数
    
    learning_rate=2e-5,
    warmup_ratio=0.1,     # 增加warmup(前10%steps线性增加lr)
    weight_decay=0.01,    # 添加正则化(防止过拟合)
    lr_scheduler_type="cosine",  # 余弦退火学习率衰减(何时需要？)

    evaluation_strategy="steps",
    eval_steps=50,  # 每50步评估一次

    save_strategy="steps",  # 按步数保存，与评估策略一致
    save_steps=100,  # 每100步保存一次模型
    save_total_limit=5,  # 设置最大保存检查点数，避免磁盘爆炸

    load_best_model_at_end=True,  # 训练结束加载最佳模型
    metric_for_best_model="f1",   # 根据F1选择最佳(mrpc任务核心指标)
    greater_is_better=True,  # 评估指标是F1，F1越大越好

    #bf16=torch.cuda.is_bf16_supported(),  # 自动检测是否支持bf16
    #fp16=not torch.cuda.is_bf16_supported(),  # 自动检测是否支持fp16
    dataloader_num_workers=4 if torch.cuda.is_available() else 2,  # 多线程加载数据(按照CPU核心数制定)
    dataloader_pin_memory=True,  # 锁定内存(加速数据加载)，建议GPU训练时开启

    logging_dir=f"{save_path}/{checkpoint}/logs",  # 日志单独存放
    logging_steps=50,  # 每50步打印一次日志
    
    # report_to=["tensorboard", "wandb"],  # 多平台监控
    report_to="tensorboard",
    save_safetensors=True,  # 启用安全格式

    # do_train=True,  # 训练开关
    # max_steps=15000,  # 总训练步数
    )

# 定义评估指标函数
def compute_metrics(eval_preds):
    metric = load("glue", "mrpc")  # 或者 load("glue/mrpc")
    
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    
    return metric.compute(predictions=predictions, references=labels)


# 创建训练器，将创建的所有对象传入进行微调
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,  # 评估指标函数
)

# 开始训练
# 注：resume_from_checkpoint=False表示从头开始训练(初次训练可选)
# 启动训练时自动检测已有checkpoint(方便训练中断后重启)
trainer.train(resume_from_checkpoint=True \
            if os.path.exists(os.path.join(save_path, checkpoint)) else False)

# 最终的保存模型
final_path = f"{save_path}/{checkpoint}_final"
trainer.save_model(final_path)  # 模型权重
tokenizer.save_pretrained(final_path)  # tokenizer
