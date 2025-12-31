import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter

# ======================================================================
# 1. CORE TRANSFORMER ARCHITECTURE
# ======================================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = F.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_probs, V)
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        return self.W_o(self.combine_heads(attn_output))

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, 
                 num_layers, d_ff, max_seq_length, dropout):
        super().__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask.to(tgt.device)
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        return self.fc(dec_output)

# ======================================================================
# 2. DATA PROCESSING & TOKENIZATION
# ======================================================================

class SimpleTokenizer:
    """Basic character/word level tokenizer for educational purposes."""
    def __init__(self, min_freq=2):
        self.vocab = {"[PAD]": 0, "[SOS]": 1, "[EOS]": 2, "[UNK]": 3}
        self.index_to_vocab = {v: k for k, v in self.vocab.items()}
        self.min_freq = min_freq

    def build_vocab(self, sentences):
        counts = Counter()
        for s in sentences:
            counts.update(s.split())
        for word, freq in counts.items():
            if freq >= self.min_freq:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
        self.index_to_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        return [self.vocab.get(w, self.vocab["[UNK]"]) for w in text.split()]

    def decode(self, indices):
        return " ".join([self.index_to_vocab.get(i, "[UNK]") for i in indices if i > 2])

class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, src_tokenizer, tgt_tokenizer):
        with open(src_file, 'r', encoding='utf-8') as f:
            self.src_data = f.readlines()
        with open(tgt_file, 'r', encoding='utf-8') as f:
            self.tgt_data = f.readlines()
        
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        src_text = self.src_data[idx].strip()
        tgt_text = self.tgt_data[idx].strip()
        
        src_tokens = self.src_tokenizer.encode(src_text)
        # Add SOS and EOS to target
        tgt_tokens = [1] + self.tgt_tokenizer.encode(tgt_text) + [2]
        
        return torch.tensor(src_tokens), torch.tensor(tgt_tokens)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_batch, tgt_batch

# ======================================================================
# 3. TRAINING AND INFERENCE LOGIC
# ======================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        optimizer.zero_grad()
        logits = model(src, tgt_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def translate(model, sentence, src_tokenizer, tgt_tokenizer, device, max_len=50):
    model.eval()
    src_tokens = torch.tensor(src_tokenizer.encode(sentence)).unsqueeze(0).to(device)
    src_mask = (src_tokens != 0).unsqueeze(1).unsqueeze(2)
    
    with torch.no_grad():
        src_emb = model.dropout(model.positional_encoding(model.encoder_embedding(src_tokens)))
        enc_out = src_emb
        for layer in model.encoder_layers:
            enc_out = layer(enc_out, src_mask)

    tgt_indices = [1] # [SOS]
    for _ in range(max_len):
        tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(device)
        _, tgt_mask = model.generate_mask(src_tokens, tgt_tensor)
        
        with torch.no_grad():
            tgt_emb = model.dropout(model.positional_encoding(model.decoder_embedding(tgt_tensor)))
            dec_out = tgt_emb
            for layer in model.decoder_layers:
                dec_out = layer(dec_out, enc_out, src_mask, tgt_mask)
            
            output = model.fc(dec_out)
            next_word = output.argmax(2)[:, -1].item()
            
        tgt_indices.append(next_word)
        if next_word == 2: # [EOS]
            break
            
    return tgt_tokenizer.decode(tgt_indices)

# ======================================================================
# 4. MAIN EXECUTION (SETUP & RUN)
# ======================================================================

if __name__ == "__main__":
    # 1. Hyperparameters
    D_MODEL = 256
    NUM_HEADS = 8
    NUM_LAYERS = 4
    D_FF = 512
    MAX_SEQ_LENGTH = 100
    DROPOUT = 0.1
    BATCH_SIZE = 32
    EPOCHS = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Check for data files
    if not (os.path.exists('train.en') and os.path.exists('train.ne')):
        print("Error: train.en and train.ne not found. Please create them first.")
        # Dummy content for testing if you don't have files:
        # with open('train.en', 'w') as f: f.write("Hello\nHow are you?")
        # with open('train.ne', 'w') as f: f.write("नमस्ते\nतपाईं कस्तो हुनुहुन्छ?")
        exit()

    # 3. Build Tokenizers
    print("Building vocabularies...")
    src_tok = SimpleTokenizer()
    tgt_tok = SimpleTokenizer()
    with open('train.en', 'r', encoding='utf-8') as f: src_tok.build_vocab(f.readlines())
    with open('train.ne', 'r', encoding='utf-8') as f: tgt_tok.build_vocab(f.readlines())

    # 4. Setup Data
    dataset = TranslationDataset('train.en', 'train.ne', src_tok, tgt_tok)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # 5. Initialize Model
    model = Transformer(
        len(src_tok.vocab), len(tgt_tok.vocab), 
        D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, MAX_SEQ_LENGTH, DROPOUT
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 6. Training Loop
    print(f"Starting training on {DEVICE}...")
    for epoch in range(EPOCHS):
        avg_loss = train_epoch(model, dataloader, optimizer, criterion, DEVICE)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    # 7. Test Translation
    test_sentence = "Hello"
    result = translate(model, test_sentence, src_tok, tgt_tok, DEVICE)
    print(f"English: {test_sentence}")
    print(f"Nepali (Translated): {result}")