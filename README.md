Developed a custom Transformer-based Language Model (TinyGPT) from scratch using PyTorch, implementing core architectural components including Multi-Head Self-Attention, Positional Embeddings, and Layer Normalization to capture long-range dependencies in text.
Engineered an end-to-end NLP training pipeline, creating a custom word-level tokenizer and dynamic batch generation system, while optimizing model convergence using the AdamW optimizer and Cross-Entropy loss over 1,500 training
epochs.

Built an interactive Generative AI application for next-word prediction, utilizing autoregressive decoding and multinomial sampling to generate coherent text sequences based on user-provided context.


This code implements a Next Word Predictor (specifically, a small Language Model) using a Transformer architecture.

Here is a breakdown of why it is a next-word predictor:

Data Preparation: It takes a text corpus, splits it into words, and converts them into numerical indices.
Training Objective: The get_batch function creates pairs of inputs (x) and targets (y).
If x is ["the", "cat", "sat"]
y is ["cat", "sat", "on"]
The model learns that after "the", the next word is "cat", and so on.
Model Architecture: The TinyGPT class uses embeddings and Transformer blocks to understand the context of previous words to predict the probability of the next word.
Generation: The generate function takes a starting word (context), predicts the next word, appends it to the sequence, and repeats the process to generate a sentence.
It is a miniature version of how large models like GPT
