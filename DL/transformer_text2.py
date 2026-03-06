1. THE CORE PROBLEM TRANSFORMERS SOLVE
Before transformers, models used:
• RNN
 • LSTM
 • GRU
Those process text token by token sequentially.
Example:
The → cat → sat → on → mat
Each step waits for the previous step.
Problems:
Slow (cannot parallelize)


Long memory problem


Information fades over distance


Example:
The cat that chased the mouse that stole the cheese sat on the mat.
The model must remember "cat" until "sat".
RNN memory leaks.
Transformers solve this by allowing every token to directly look at every other token.
No sequential bottleneck.
This mechanism is called:
SELF ATTENTION
This is the heart of the transformer.

2. TOKENIZATION
First the sentence must become numbers.
Example:
"The cat sat"
Tokenized:
["The", "cat", "sat"]
Then mapped to IDs:
[101, 5423, 923]
Vocabulary size may be:
50,000
Now the model can process them.
But integers themselves mean nothing.
So we convert them into vectors.

3. EMBEDDINGS
Each token becomes a vector.
Example:
"The" → [0.21, -0.44, 0.91, ...]
Dimension might be:
512
So sentence becomes matrix:
Sequence length = 3
Embedding size = 512
Matrix shape:
3 × 512
Think of embeddings as coordinates in meaning space.
Words with similar meaning end up nearby.
Example:
king ≈ queen
cat ≈ dog
Paris ≈ London
But we have a problem.
Transformers do not know order.

4. POSITIONAL ENCODING
Self-attention treats tokens like a bag of words.
dog bites man
man bites dog
Same tokens.
Different meaning.
So we inject position information.
Positional encoding is a vector added to embeddings.
Example:
Embedding(word) + PositionVector
Original embedding:
[0.3, 0.7, -0.2]
Position vector:
[0.1, 0.2, 0.4]
Result:
[0.4, 0.9, 0.2]
Classic transformer uses sinusoidal encoding.
Formulas:
PE(pos,2i) = sin(pos / 10000^(2i/d_model))
PE(pos,2i+1) = cos(pos / 10000^(2i/d_model))
Why sine/cosine?
Because they create smooth periodic patterns that allow the model to infer relative positions.
Now each token has:
word meaning + position

5. ENTER THE TRANSFORMER BLOCK
A transformer is built from stacked blocks.
Typical architecture:
Embedding
↓
Positional Encoding
↓
Transformer Block
↓
Transformer Block
↓
Transformer Block
↓
Output Layer
Each block has two major components:
1. Self Attention
2. Feed Forward Network
Plus:
Residual connections
Layer normalization
Let’s open the first major mechanism.

6. SELF ATTENTION
The idea:
Each word asks:
"Which other words are important for understanding me?"
Example sentence:
The animal didn't cross the street because it was tired
Word "it" needs to attend to animal.
Self-attention allows that.
Mechanism uses three vectors:
Query
Key
Value

7. QUERY, KEY, VALUE
For each token we create three vectors.
From embedding:
X
We multiply by three weight matrices.
Q = XWq
K = XWk
V = XWv
Shapes example:
X : (seq_len × d_model)

Wq : (d_model × d_k)
Wk : (d_model × d_k)
Wv : (d_model × d_v)
So each token now has:
Query vector
Key vector
Value vector
Interpretation:
Query = what I'm looking for
 Key = what I offer
 Value = my information

8. ATTENTION SCORES
Now we compare queries and keys.
Formula:
score = Q × Kᵀ
This produces:
(seq_len × seq_len)
Example:
3 tokens

attention matrix = 3 × 3
Each number represents how much token i attends to token j.
But raw scores can explode.
So we scale them.

9. SCALED DOT PRODUCT ATTENTION
Full formula:
Attention(Q,K,V) = softmax(QKᵀ / √d_k) V
Why divide by √d_k ?
Because large vectors produce huge dot products.
Scaling stabilizes gradients.
Then we apply:
softmax
Softmax turns scores into probabilities.
Example:
[2.1, 0.3, 1.4]
Softmax →
[0.60, 0.10, 0.30]
Meaning:
Token attends:
60% token1
10% token2
30% token3
Then multiply with V.
This produces the new representation of the token.
Each token now becomes a weighted mixture of other tokens.

10. MULTI HEAD ATTENTION
One attention is good.
Many attentions are better.
Transformer uses multiple heads.
Example:
8 attention heads
Each head learns different relationships.
Example:
Head 1 might learn:
subject → verb
Head 2:
pronoun → noun
Head 3:
adjective → noun
Process:
Head1 = Attention(Q1,K1,V1)
Head2 = Attention(Q2,K2,V2)
Head3 = Attention(Q3,K3,V3)
Outputs are concatenated.
Concat(head1, head2, head3...)
Then projected back using matrix:
Wo
This forms the multi-head attention output.

11. RESIDUAL CONNECTION
Transformers use skip connections.
Instead of:
output = attention(X)
We do:
output = X + attention(X)
Why?
Because deep networks suffer from:
vanishing gradients
Residual paths allow gradients to flow easily.
Think of it as a highway for information.

12. LAYER NORMALIZATION
Next we normalize values.
LayerNorm stabilizes training.
Formula conceptually:
x_norm = (x - mean) / std
This keeps activations within reasonable ranges.
Without it:
training becomes unstable.

13. FEED FORWARD NETWORK (Epochs)
After attention we apply a small neural network to each token.
Structure:
Linear
ReLU
Linear
Example sizes:
512 → 2048 → 512
So:
FFN(x) = max(0, xW1 + b1) W2 + b2
Why do this?
Attention mixes information between tokens.
Feedforward processes each token individually.
It increases model capacity.

14. FULL TRANSFORMER BLOCK
One block looks like this:
Input
↓
Multi Head Attention
↓
Add & LayerNorm
↓
Feed Forward
↓
Add & LayerNorm
↓
Output
This block is stacked many times.
Example:
GPT-3: 96 layers
BERT: 12 or 24 layers
Stacking increases reasoning depth.

15. ENCODER VS DECODER
Original transformer has two parts.
Encoder
Reads input sentence.
Input → Encoder → Context representation
Decoder
Generates output sequence.
Context → Decoder → Output tokens
Encoder blocks contain:
Self Attention
Feed Forward
Decoder blocks contain:
Masked Self Attention
Encoder-Decoder Attention
Feed Forward

16. MASKED ATTENTION
In generation we must prevent cheating.
Example:
Predicting:
The cat ___
Model must not see the future token.
So we apply causal mask.
Attention matrix becomes:
[1 0 0 0]
[1 1 0 0]
[1 1 1 0]
[1 1 1 1]
Future tokens are blocked.

17. OUTPUT LAYER
Final hidden vector passes through:
Linear projection
Mapping from hidden size to vocabulary size.
Example:
512 → 50,000
Output becomes logits:
[0.2, -1.1, 4.3, 0.7 ...]
Apply softmax:
probabilities over vocabulary
Highest probability token is chosen.

18. TRAINING
During training we know the correct next token.
Loss used:
Cross entropy
Example:
Predicted: [0.1,0.2,0.7]
True:      [0,0,1]
Loss penalizes wrong predictions.
Backpropagation updates all weights.
Including:
Embeddings
Attention matrices
Feedforward layers
Training runs over billions of tokens.

19. INFERENCE (TEXT GENERATION)
Generation loop:
Input tokens
↓
Transformer
↓
Predict next token
↓
Append token
↓
Repeat
Example:
Input: "The cat"
Prediction: "sat"
Next step input:
"The cat sat"
This continues until end token.

20. FINAL TRANSFORMER PIPELINE
Full pipeline:
Text
↓
Tokenization
↓
Embedding
↓
Positional Encoding
↓
Transformer Block × N
  • Multi Head Attention
  • Feed Forward
  • Residual
  • LayerNorm
↓
Linear Projection
↓
Softmax
↓
Next Token

Text Input
     ↓
Tokenization
     ↓
Token IDs
     ↓
Embedding Layer
     ↓
Positional Encoding
     ↓
Transformer Blocks (repeated many times)
     ↓
Self-Attention
     ↓
Feed Forward Network
     ↓
Layer Normalization + Residual Connections
     ↓
Final Linear Layer
     ↓
Softmax
     ↓
Next Token Prediction


"""

Read this paper : "https://arxiv.org/abs/1706.03762"