# In this folder you will find the scripts used in this project.
## Overview
Each script grabs a prompt, a json template (look in the example_template folder to see how that looks like) and an image or pdf as an input.
It outputs a json file in the same format as the template.json but with the questions answered.
* gpt_analyze_image.py, claude_analyze_image.py and gemini_analyze_image.py -> Answer questions by reading an image
*  gpt_analyze_pdf.py and claude_analyze_pdf.py-> Answer questions by reading a pdf document instead of an image, only claude and GPT were used and only applied to 1 newspaper because this project was more focused on reading images.
## Usage
### GPT (image)
Setup:
    pip install openai
    export OPENAI_API_KEY=sk-...

Usage:
    python gpt_analyze_image.py newspaper.jpg
    python gpt_analyze_image.py newspaper.jpg --prompt prompt.txt --template template.json --output answer.json
### GPT (pdf)
Setup:
    pip install openai
    export OPENAI_API_KEY=sk-...

Usage:
    python gpt_analyze_pdf.py newspaper.pdf
    python gpt_analyze_pdf.py newspaper.pdf --prompt prompt.txt --template template.json --output answer.json
### Claude (image) 
Setup:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...

Usage:
    python claude_analyze_image.py newspaper.jpg
    python claude_analyze_image.py newspaper.jpg --prompt prompt.txt --template template.json --output answer.json
### Claude (pdf)
Setup:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...

Usage:
    python claude_analyze_pdf.py newspaper.pdf
    python claude_analyze_pdf.py newspaper.pdf --prompt prompt.txt --template template.json --output answer.json
