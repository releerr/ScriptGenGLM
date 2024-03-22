
# scriptGenGLM1.0

模型用于中文剧本生成，用户通过与模型交互生成剧本。


## Model Details

### Model Description

用户与模型交互例子：
用户输入：人物简介：关羽：正直善良的性格。母亲：坚强不妥协的妇人家。父亲：为国牺牲的好男儿。
    剧情大纲：第一集：xxxx:讲述在xxxxxx。



- **Developed by:** Xinrui Li(Relee)
- **Language(s) (NLP):** Chinese
- **License:** Apache License Version 2.0
- **Finetuned from model:** chatglm3-6b

### Model Sources [optional]

- **Repository:** https://github.com/releerr/ScriptGenGLM.git


### Direct Use
**请注意，该模型基于chatglm3-6b模型进行微调，我们并没有合并训练后的模型，而是在adapter_config.json中记录了微调型的路径。您需要首先部署好chatglm3-6b模型后，将adapter_config.json中base_model_name_or_path的路径修改为您部署chatglm3-6b模型的路径。**

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("outputNOEnter/checkpoint-3000", trust_remote_code=True)
model = AutoModel.from_pretrained("outputNOEnter/checkpoint-3000", trust_remote_code=True, device='cuda')
model = model.eval()
response, history = model.chat(tokenizer, "你好,你可以根据我提供的信息以Markdown格式写一集剧本吗？要求300字以内。", history=[])
print("\n\n"+response)
response, history = model.chat(tokenizer, "人物简介：我（孟刚）：战火中成长，后成为团长为国奉献牺。母亲：坚强不妥协的妇人家。父亲：为国牺牲的好男儿。桃子：我的妻子，坚守家园，等待我（孟刚）的归来。剧情大纲：第一集：你带我回家: 讲述在抗美援朝时期，孟刚的父亲在前线牺牲，母亲在废墟中悲痛欲绝，带着年幼的孟刚踏上了归途。", history=[])
print("\n\n"+response)


### Recommendations


Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.



## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).



### Framework versions

- PEFT 0.10.0
