from huggingface_hub import login
import wandb


"""
第1步：初始化设置和登录

设置访问令牌并登录到HuggingFace和Weights&Biases平台
"""
# 直接设置访问令牌
hf_token = "XXX"  # 替换为你的token
wb_token = "XXX"        # 替换为你的token

# 验证token是否存在并有效
if not hf_token or hf_token.strip() == "":
    raise ValueError("""
    未找到有效的HUGGINGFACE_TOKEN。
    请确保已设置正确的HuggingFace token。
    可以从 https://huggingface.co/settings/tokens 获取新的token
    """)

try:
    # 登录HuggingFace
    login(token=hf_token, write_permission=True)
    print("HuggingFace登录成功!")
    
    # 登录Weights & Biases
    if wb_token:
        wandb.login(key=wb_token)
        print("Weights & Biases登录成功!")
    else:
        print("警告: 未设置WANDB_TOKEN，将使用匿名模式")
        
except Exception as e:
    print(f"登录失败: {str(e)}")
    print("\n请确保:")
    print("1. token格式正确 (应该以 'hf_' 开头)")
    print("2. token具有足够的权限 (需要 'write' 权限)")
    print("3. token未过期")
    raise

# 初始化wandb项目
run = wandb.init(
    # 项目名称
    project='Fine-tune-DeepSeek-R1-Distill-Qwen-1.5B',
    # 实验类型
    job_type="training",
    # 匿名设置,allow表示允许匿名访问实验结果
    # 可选值:
    # - allow: 允许匿名访问
    # - must: 必须匿名
    # - never: 不允许匿名
    anonymous="allow"
)


"""
第2步：加载模型和分词器

使用unsloth优化的FastLanguageModel加载预训练模型
"""
from unsloth import FastLanguageModel

# 模型配置参数
max_seq_length = 2048  # 最大序列长度
dtype = None          # 数据类型，None表示自动选择
load_in_4bit = True   # 使用4bit量化加载模型以节省显存


# 加载预训练模型和分词器
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = hf_token, 
)


"""
第3步：定义提示模板和进行微调前的推理测试
"""
prompt_style = """以下是描述任务的指令，以及提供更多上下文的输入。
请写出恰当完成该请求的回答。
在回答之前，请仔细思考问题，并创建一个逐步的思维链，以确保回答合乎逻辑且准确。

### Instruction:
你是一位在临床推理、诊断和治疗计划方面具有专业知识的医学专家。
请回答以下医学问题。

### Question:
{}

### Response:
<think>{}"""

# 测试用医学问题
question = "宝宝病毒感染，高烧38.6,吃上退烧药就好，停了就又发烧，请问像这种病毒性感冒发烧吃什么药好？"

# 设置模型为推理模式
FastLanguageModel.for_inference(model) 
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

# 生成回答
outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=1200,
    use_cache=True,
)
response = tokenizer.batch_decode(outputs)
print("### 微调前模型推理结果：")
print(response[0].split("### Response:")[1])


"""
第4步：数据集处理函数
"""
train_prompt_style = """以下是描述任务的指令，以及提供更多上下文的输入。
                        请写出恰当完成该请求的回答。
                        在回答之前，请仔细思考问题，并创建一个逐步的思维链，以确保回答合乎逻辑且准确。

                        ### Instruction:
                        你是一位在临床推理、诊断和治疗计划方面具有专业知识的医学专家。
                        请回答以下医学问题。

                        ### Question:
                        {}

                        ### Response:
                        <think>
                        {}
                        </think>
                        {}"""

EOS_TOKEN = tokenizer.eos_token  # 添加结束符标记

#格式化提示函数,用于处理数据集中的示例
def formatting_prompts_func(examples):
    # 从examples中提取问题、思维链和回答
    inputs = examples["Question"]      # 医学问题列表
    cots = examples["Complex_CoT"]     # 思维链列表 
    outputs = examples["Response"]     # 回答列表
    
    # 存储格式化后的文本
    texts = []

    # 遍历每个示例,将问题、思维链和回答组合成指定格式
    for input, cot, output in zip(inputs, cots, outputs):
        # 使用train_prompt_style模板格式化文本,并添加结束符
        text = train_prompt_style.format(input, cot, output) + EOS_TOKEN
        texts.append(text)
        
    # 返回格式化后的文本字典
    return {
        "text": texts,
    }

# 加载数据集并应用格式化
from datasets import load_dataset
dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT","zh", split = "train[0:500]",trust_remote_code=True)
dataset = dataset.map(formatting_prompts_func, batched = True,)


"""
第5步：配置LoRA微调参数

使用LoRA技术进行参数高效微调
"""
FastLanguageModel.for_training(model)

model = FastLanguageModel.get_peft_model(
    # 原始模型
    model, 
    # LoRA秩,用于控制低秩矩阵的维度,值越大表示可训练参数越多,模型性能可能更好但训练开销更大
    # 建议: 8-32之间
    r=16,  
    # 需要应用LoRA的目标模块列表
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention相关层
        "gate_proj", "up_proj", "down_proj",     # FFN相关层
    ],
    # LoRA缩放因子,用于控制LoRA更新的幅度。值越大，LoRA的更新影响越大。
    lora_alpha=16,
    # LoRA层的dropout率,用于防止过拟合,这里设为0表示不使用dropout。
    # 如果数据集较小，建议设置0.1左右。
    lora_dropout=0,  
    # 是否对bias参数进行微调,none表示不微调bias
    # none: 不微调偏置参数；
    # all: 微调所有参数；
    # lora_only: 只微调LoRA参数。
    bias="none",  
    # 是否使用梯度检查点技术节省显存,使用unsloth优化版本
    # 会略微降低训练速度，但可以显著减少显存使用
    use_gradient_checkpointing="unsloth", 
    # 随机数种子,用于结果复现
    random_state=0,
    # 是否使用rank-stabilized LoRA,这里不使用
    # 会略微降低训练速度，但可以显著减少显存使用
    use_rslora=False,  
    # LoFTQ配置,这里不使用该量化技术，用于进一步压缩模型大小
    loftq_config=None,
)


"""
第6步：配置训练参数和初始化训练器
"""
from trl import SFTTrainer  # 用于监督微调的训练器
from transformers import TrainingArguments  # 用于配置训练参数
from unsloth import is_bfloat16_supported  # 检查是否支持bfloat16精度训练

# 初始化SFT训练器
trainer = SFTTrainer(
    model=model,  # 待训练的模型
    tokenizer=tokenizer,  # 分词器
    train_dataset=dataset,  # 训练数据集
    dataset_text_field="text",  # 数据集字段的名称
    max_seq_length=max_seq_length,  # 最大序列长度
    dataset_num_proc=2,  # 数据集处理的并行进程数，提高CPU利用率
    args=TrainingArguments(
        per_device_train_batch_size=1,  # 每个GPU的训练批次大小
        gradient_accumulation_steps=4,   # 梯度累积步数,用于模拟更大的batch size
        warmup_steps=5,  # 预热步数,逐步增加学习率
        learning_rate=2e-4,  # 学习率
        lr_scheduler_type="linear",  # 线性学习率调度器
        max_steps=60,    # 最大训练步数（一步 = 处理一个batch的数据）
        # 根据硬件支持选择训练精度
        fp16=False,  # 禁用混合精度训练
        bf16=True,      # 启用BF16
        logging_steps=10,  # 每10步记录一次日志
        optim="adamw_8bit",  # 使用8位AdamW优化器节省显存，几乎不影响训练效果
        weight_decay=0.01,   # 权重衰减系数,用于正则化，防止过拟合
        seed=3407,  # 随机数种子
        output_dir="outputs",  # 保存模型检查点和训练日志
    ),
)


"""
第7步 开始训练
"""
trainer.train()


"""
第8步：微调后的模型推理测试
"""
question = "宝宝病毒感染，高烧38.6,吃上退烧药就好，停了就又发烧，请问像这种病毒性感冒发烧吃什么药好？"

# 启用模型推理模式,使用Unsloth加速推理速度
FastLanguageModel.for_inference(model)  

# 对输入问题进行编码,转换为模型可处理的张量格式并移至GPU
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

# 生成回答
outputs = model.generate(
    input_ids=inputs.input_ids, # 输入token的id序列
    attention_mask=inputs.attention_mask,  # 注意力掩码,用于标记有效输入位置
    max_new_tokens=1200, # 生成的最大新token数量
    use_cache=True, # 是否使用KV缓存加速生成
)

# 解码模型输出
response = tokenizer.batch_decode(outputs)
print("### 微调后模型推理结果：")
print(response[0].split("### Response:")[1])


"""
第9步：保存模型

包括保存完整模型和合并后的模型
"""
new_model_local = "DeepSeek-R1-Medical-COT-Qwen-1.5B"
model.save_pretrained(new_model_local) 
tokenizer.save_pretrained(new_model_local)

# 保存合并后的16bit模型
model.save_pretrained_merged(new_model_local, tokenizer, save_method = "merged_16bit",)


"""
第10步：模型上传代码
"""
# 定义在线仓库地址 Your_HuggingFace_Name为你HuggingFace的用户名称
new_model_online = "Your_HuggingFace_Name/DeepSeek-R1-Medical-COT-Qwen-1.5B"
# 上传LoRA权重和配置
model.push_to_hub(new_model_online)
# 上传分词器    
tokenizer.push_to_hub(new_model_online)
# 上传合并后的16bit模型
model.push_to_hub_merged(new_model_online, tokenizer, save_method = "merged_16bit")
