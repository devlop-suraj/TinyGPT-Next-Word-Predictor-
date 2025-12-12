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


graph TD
    Input[Input Indices (B, T)] --> TokenEmb[Token Embedding (vocab_size, 32)]
    Input --> PosEmb[Positional Embedding (block_size, 32)]
    
    TokenEmb --> Sum((+))
    PosEmb --> Sum
    
    Sum --> Block1[Transformer Block 1]
    Block1 --> Block2[Transformer Block 2]
    
    Block2 --> LN_F[Layer Norm Final]
    LN_F --> Head[Linear Head (32 -> vocab_size)]
    
    Head --> Logits[Logits (B, T, vocab_size)]
    Logits --> Softmax[Softmax]
    Softmax --> Output[Next Word Probabilities]



    exactly what is happening inside your model based on your variables:

Inputs (idx)

Shape: (Batch_Size, Time_Steps) e.g., (16, 6)
These are integers representing words (e.g., word2idx['cat']).
Embeddings Layer

Token Embedding: Converts word indices to vectors.
Size: vocab_size 
×
× 32 (embedding_dim)
Positional Embedding: Learnable vectors representing the position of a word in the sequence (0 to 5).
Size: 6 (block_size) 
×
× 32
Combination: These two are added together element-wise.
Transformer Blocks (self.blocks)

You have 2 Layers (n_layers = 2).
Inside each Block (inferred from standard implementation):
Layer Norm 1
Multi-Head Self-Attention:
n_heads = 2
Each head has size 32 / 2 = 16.
This allows the model to look at previous words to understand context.
Layer Norm 2
Feed-Forward Network (MLP): A small neural network that processes the information.
Residual Connections: The input is added to the output of the attention and MLP layers to help training.
Final Output Layer

Layer Norm (ln_f): Stabilizes the output before the final prediction.
Linear Head (head): Projects the 32-dimensional vector back up to the size of your vocabulary (vocab_size).
Output: Raw scores (logits) for every possible word in your dictionary.
Summary of Hyperparameters
Context Window (block_size): 6 words (The model only looks at the last 6 words).
Model Width (embedding_dim):** 32 (Each word is represented by 32 numbers).

Depth (n_layers): 2 Transformer blocks.
Heads (n_heads): 2 Attention heads.
