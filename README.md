---
base_model: models/qwen2.5-14B-instruct
library_name: peft
language:
  - zh
frameworks: PyTorch
base_model_relation: finetune
license: Apache License 2.0
tasks:
  - text-generation
  - task-oriented-conversation
  - question-answering
  - script generation 
---
# SageGen

- 模型用于中文古装类型短剧剧本生成，用户通过与模型交互生成剧本。
- modelscope: [notnot/ScriptGen] (https://modelscope.cn/models/notnot/ScriptGen)
- huggingFace: [Releer/Sage_AncientCostumeShortPlayScriptGenerator] https://huggingface.co/Releer/Sage_AncientCostumeShortPlayScriptGenerator



## 古装剧本生成器

### Model Description

- 用户可以通过输入简单的流水账故事情节内容，模型将会基于用户给到的故事情节基础之上进行改写并扩写成一个完整的古装短剧剧本。 



- **Developed by:** Xinrui Li (Relee)
- **Language(s) (NLP):** 中文
- **License:** Apache License Version 2.0
- **Finetuned from model:** Qwen2.5-14B-Instruct



## Uses


### Direct Use

**请注意，该模型基于Qwen2.5-14B-Instruct模型进行微调，我们并没有合并训练后的模型，而是在adapter_config.json中记录了微调型的路径。您需要首先部署好Qwen2.5-14B-Instruct模型后，将adapter_config.json中base_model_name_or_path的路径修改为您部署Qwen2.5-14B-Instruct模型的路径。**
- Qwen/Qwen2.5-14B-Instruct: https://modelscope.cn/models/Qwen/Qwen2.5-14B-Instruct

#快速使用:
---------------------------------------------------------------------------------
```
from modelscope import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
import torch
import os

modelName = "SageGen_qwen2.5_14B_instruct"
accelerator = Accelerator()

tokenizers = AutoTokenizer.from_pretrained(modelName,trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    modelName,
    torch_dtype='auto',
    device_map='auto',
    trust_remote_code=True
)

messages = [
    {"role": "system", "content": system_example},
    {"role": "user", "content": user_example},
    {"role": "assistant", "content": assistance_example}

]

userInput_query = input("用户描述内容为:")

messages.append({"role":"user","content":userInput_query})

user_inputs = tokenizers.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
    return_dict=True
)
model, user_inputs = accelerator.prepare(model, user_inputs)
user_inputs = user_inputs.to(model.device)

with torch.no_grad():
    print("开始推理\n\n")
    torch.cuda.empty_cache()  # 如果使用GPU
    generated_ids = model.generate(
        **user_inputs,
        max_new_tokens=20000,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.1, 

    )

generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(user_inputs.input_ids, generated_ids)]
response = tokenizers.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)

```
---------------------------------------------------------------------------------  
- PEFT 0.14.0
