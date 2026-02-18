Mini Instruct-GPT
------

Mini Instruct-GPT is a lightweight, end‑to‑end framework for training and serving small instruction‑tuned language models.
It is designed to be simple, modular, and easy to extend, so you can quickly experiment with data, architectures, and training setups without heavyweight infrastructure.
The project covers the full pipeline from data preparation and formatting, through supervised fine‑tuning, to inference and evaluation, making it a practical starting point for learning, prototyping, or building custom instruction‑following models on limited compute.


Files
-----
**main.ipynb:** Jupyter notebook that implements the end‑to‑end pipeline, from loading the Dolly‑15k dataset through supervised fine‑tuning and PPO‑based RLHF training of DeepSeek-R1-Distill-Qwen-1.5B, plus basic inference examples.

**dolly15k-train_with_responses.csv:** Preprocessed subset of the Dolly‑15k instruction‑following dataset, with prompts and response pairs formatted for SFT and RLHF training in the notebook pipeline.