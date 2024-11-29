## CC-Time


## Requirements
To install all dependencies:
```
pip install -r requirements.txt
```

## Datasets
You can access the well pre-processed datasets from [Google Drive](https://drive.google.com/file/d/1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP/view), then place the downloaded contents under ./dataset
## Quick Demos
1. Download datasets and place them under ./datasets
2. Download the large language models from [Hugging Face](https://huggingface.co/). 
* [GPT2](https://huggingface.co/openai-community/gpt2)
3. Use ./models/hugging_gpt2/modeling_gpt2.py to replace the GPT2 architecture modeling_gpt2.py in the transformer library
4. Run main.py
