# Module 1 — LLM Fundamentals

Deep-dives on the pieces the base repo glosses over. The existing `src/` code is
the baseline; M1 adds the *why* and the modern variants that become defaults for
every later module.

| Sub-module | Files | Status |
|---|---|---|
| 1.1 Tokenizers from scratch | `bpe_tokenizer.py`, `tokenizer_comparison.py`, `test_bpe_tokenizer.py` | ✅ |
| 1.2 Positional encodings | `src/models/rope.py`, `positional_encodings.py`, `test_positional_encodings.py` | ✅ |
| 1.3 Modern block components | `src/models/norms.py`, `feedforward.py` (SwiGLU), `block_components.py`, `test_block_components.py` | ✅ |
| 1.4 Attention anatomy | `attention_deep_dive.py`, `test_attention_anatomy.py` | ✅ |

Run everything:

```bash
# correctness first (repo golden rule), then the benchmark
python -m pytest modules/m1_fundamentals/ -v
python modules/m1_fundamentals/tokenizer_comparison.py            # fast (CI)
python modules/m1_fundamentals/tokenizer_comparison.py --full     # full sweep + sentencepiece
```

---

## 1.1 Tokenizers from scratch (byte-level BPE)

`bpe_tokenizer.py` implements the tokenizer the GPT family actually uses:
**byte-level Byte-Pair Encoding with GPT-2 regex pre-tokenization**. It's a
drop-in alternative to the repo's char-level `SimpleTokenizer` (wire it in via
`prepare_data(text_file, tokenizer=trained_bpe)`).

### How BPE works

1. **Start from bytes.** Encode text as raw UTF-8 bytes → base vocabulary of
   exactly **256** symbols. Any Unicode string is representable, so there is
   *no OOV and no `<UNK>` token*, ever.
2. **Pre-tokenize with a regex.** Split text into chunks (words, number runs,
   punctuation runs) using GPT-2's pattern. Merges are only learned *within* a
   chunk — never across a space/word boundary.
3. **Iteratively merge the most frequent adjacent pair.** Each merge invents a
   new token id and records the rule `(a, b) -> new_id`. Repeat until the vocab
   reaches the target size.
4. **Encode** replays merges in learned order; **decode** concatenates the byte
   string each id expands to.

### The vocab-size ↔ sequence-length tradeoff

This is the whole point of a tokenizer, and the benchmark measures it directly.
On `data/sample_text.txt`:

| tokenizer | vocab | tokens | bytes/token | seq len vs char |
|---|---:|---:|---:|---:|
| char | 34 | 13400 | 1.00 | 1.00 |
| bpe-512 | 512 | 2960 | 4.53 | 0.22 |
| bpe-1024 | 543* | 2341 | 5.72 | 0.18 |

\* the small sample corpus runs out of distinct pairs before reaching 1024 —
you can't fill a large vocab from a tiny corpus (an instructive result in
itself). BPE cuts sequence length to ~1/5 of char-level, and since attention is
**O(n²)** in sequence length, that's a direct compute win. Larger vocab → fewer
tokens per byte → shorter sequences, but a bigger embedding/softmax matrix and
rarer tokens that are harder to learn. Frontier models land around 32k–256k.

### Classic gotchas (interview gold)

- **Whitespace handling.** GPT-2 glues a leading space to the following word
  (`" ?\p{L}+"`), so `"dog"` and `" dog"` are *different tokens*. This is why
  prompts sometimes behave differently with/without a trailing space, and why
  the first word of a document tokenizes oddly.
- **Number tokenization.** Digits merge greedily by frequency, so `"1234"` might
  be one token but `"1235"` three. Inconsistent digit chunking is a real source
  of arithmetic errors — GPT-2/3 are notoriously bad at math partly for this
  reason; some newer tokenizers force single-digit tokens.
- **"Why can't it spell strawberry?"** The model never sees characters — it sees
  `"straw"` + `"berry"` (or similar). Counting the r's requires reasoning about
  sub-word pieces it can't see inside. Tokenization, not intelligence, is the
  bottleneck for character-level tasks.
- **Byte fallback = no OOV.** Because the base is 256 bytes, a never-before-seen
  emoji or script still encodes (as its raw bytes) and round-trips losslessly.
  Contrast word-level tokenizers, which need an explicit `<UNK>`.
- **Case & morphology.** `"Dog"`, `"dog"`, `"DOG"` are distinct tokens; the
  model has to learn they're related. This inflates vocab and is why some
  tokenizers experimented with case-folding markers.

### Interview questions

1. **What problem does BPE solve that word- and char-level tokenizers don't?**
   Word-level has a huge, brittle vocab and an OOV problem; char-level has a tiny
   vocab but very long sequences (expensive O(n²) attention). BPE interpolates:
   frequent words → single tokens, rare words → sub-word pieces, byte base → no
   OOV.
2. **Why *byte*-level BPE instead of Unicode-character-level BPE?** A fixed base
   of 256 covers every possible input; character-level would need the base vocab
   to include every Unicode codepoint (~150k) and still hit unseen ones.
3. **What does the regex pre-tokenization buy you?** It prevents merges across
   semantic boundaries (e.g. merging a word with the following punctuation or
   across spaces), keeps tokens interpretable, and bounds the search.
4. **Walk through `decode(encode(s)) == s`. Why is it guaranteed?** Encoding maps
   bytes→ids via reversible merges; decoding expands each id back to its exact
   byte string and concatenates. No information is discarded, so it's lossless
   for any UTF-8 input.
5. **During encoding, in what order are merges applied and why does order
   matter?** Greedily apply the merge with the *smallest id* (earliest learned)
   available, repeatedly. This reconstructs training-time order; applying merges
   in the wrong order yields a different (still-valid-looking) but inconsistent
   tokenization.
6. **How does vocab size affect model size and speed?** Bigger vocab → bigger
   embedding + output-projection matrices (vocab × d_model) and a bigger softmax,
   but shorter sequences (cheaper attention). It's a compute/memory tradeoff, not
   free.
7. **Why is GPT bad at arithmetic and spelling, tokenization-wise?** Numbers
   chunk inconsistently and words hide their characters inside multi-char tokens,
   so per-character operations aren't directly observable to the model.
8. **What's the difference between BPE, WordPiece, and Unigram (SentencePiece)?**
   BPE merges by frequency; WordPiece merges by likelihood gain; Unigram starts
   from a big vocab and prunes by a probabilistic objective. SentencePiece is a
   framework that implements BPE/Unigram directly on raw text (treats space as a
   symbol `▁`), avoiding language-specific pre-tokenization.
9. **How do special tokens (`<|endoftext|>`, chat-role markers) fit in?** They're
   reserved ids above the merged vocab, matched verbatim before BPE runs, and
   never participate in merges. (Used by M5 chat templates.)
10. **What is "token healing" / why can a trailing space break generation?**
    A prompt ending mid-token (e.g. a bare space) forces the model to start from
    an unnatural token boundary; token healing re-tokenizes the boundary to the
    most likely completion.
11. **How would you extend a trained tokenizer's vocab without retraining?**
    Append new merges/specials at higher ids; existing ids are preserved so old
    checkpoints stay valid — but new tokens have untrained embeddings.
12. **Why do multilingual models need larger vocabs?** Non-Latin scripts encode
    to more bytes per character, so without vocab coverage they fragment into
    many tokens ("token tax"), inflating cost and sequence length for those
    languages.

---

## 1.2 Positional encodings (learned · sinusoidal · RoPE)

Attention is permutation-equivariant — without a position signal, "dog bites
man" and "man bites dog" are identical to the model. `src/models/rope.py` is the
production core (wired into `GPTConfig(pos_encoding=...)`);
`positional_encodings.py` has the annotated reference derivations and the
extrapolation experiment.

```bash
python modules/m1_fundamentals/positional_encodings.py           # fast (CI)
python modules/m1_fundamentals/positional_encodings.py --full     # real training
```

### The three schemes

| Scheme | How | Params | Extrapolates? |
|---|---|---|---|
| **Learned** (GPT-2) | trainable vector per position, added to token emb | `max_seq_len × d_model` | ❌ no embedding past `max_seq_len` |
| **Sinusoidal** (orig. Transformer) | fixed sin/cos table, added to token emb | 0 | partially (defined everywhere, but absolute values unseen) |
| **RoPE** (LLaMA/Qwen/Mistral) | rotate Q,K by angle ∝ position, in attention | 0 | ✅ best — score depends on *relative* offset |

### Why RoPE works (the one identity that matters)

RoPE rotates each 2-D sub-space of a head's query/key by an angle
`θ = position · freqᵢ`. Rotations are norm-preserving, and crucially:

```
⟨RoPE(q, m), RoPE(k, n)⟩  =  f(q, k, m − n)
```

the attention score between a query at position `m` and a key at position `n`
depends **only on the relative offset `m − n`**, never on absolute positions.
That's why a head that learns "attend 4 tokens back" keeps working at positions
it never saw in training. `test_positional_encodings.py` verifies this
numerically (max score deviation across same-offset pairs ≈ 1e-6).

**Complex-number view:** pair dims into `z = x_even + i·x_odd`; rotating by `θ`
is multiplying by `e^{iθ}`. We implement the equivalent real "rotate_half" form
(`x·cos + rotate_half(x)·sin`) because it's faster, and test it against the
explicit rotation matrix.

### Experiment: length extrapolation (delayed-copy task)

Train tiny models at length `L_train`, evaluate at a longer `L_eval`. The task
(`target[i] = x[i − 4]` on **random** tokens) forces the model to attend a fixed
relative offset, so it can't cheat with token content. `--full` result:

| PE | loss @ train len (128) | loss @ eval len (256) | extrapolation gap |
|---|---:|---:|---:|
| learned | 0.005 | 5.04 | **+5.03** (collapses) |
| sinusoidal | 4.85 | 4.85 | flat (hard to learn at this scale) |
| **rope** | 0.005 | **2.19** | **+2.19** (best) |

RoPE **matches** learned in-distribution and **beats** it badly on
extrapolation — learned PE has no trained embedding for positions ≥ 128 so it
collapses to random. (Plot: `benchmarks/results/m1_positional_encoding.png`.)

### Context extension (for the long-context discussion later)

`RotaryEmbedding(scaling=...)` implements two ways to run a RoPE model past its
trained length:
- **`linear`** (Position Interpolation): divide positions by `scale_factor` to
  squeeze a longer sequence into the trained angle range. Cheap; needs light
  fine-tuning.
- **`ntk`** (NTK-aware): raise the RoPE base so high-frequency dims interpolate
  less — extends context with *no* fine-tuning by preserving fine resolution.

### Interview questions

1. **Why do transformers need positional encoding at all?** Self-attention is a
   set operation (permutation-equivariant); without position info word order is
   invisible.
2. **Learned vs sinusoidal — tradeoffs?** Learned is flexible but bounded to
   `max_seq_len` (no extrapolation, extra params); sinusoidal is parameter-free
   and defined for any position but is a fixed inductive bias.
3. **What is RoPE and why did it win?** Rotate Q/K by position-dependent angles
   so attention scores depend only on relative offset. No params, applied in
   attention, extrapolates well, and plays nicely with KV caching.
4. **Prove the relative-position property.** Rotating q by `mθ` and k by `nθ` and
   taking the inner product yields a function of `(m−n)θ` (rotation matrices
   compose: `R_mᵀ R_n = R_{n−m}`).
5. **Is RoPE absolute or relative?** It's *applied* absolutely (each token
   rotated by its own position) but *behaves* relatively in the score — the best
   of both.
6. **Why apply RoPE to Q and K but not V?** Q·K produces the attention scores
   (where relative position must live); V carries content that's mixed by those
   scores — rotating it would corrupt the values.
7. **What breaks when you run a model past its training length?** Learned PE has
   no embedding for new positions; sinusoidal/RoPE can run but attention
   patterns drift because the model never saw those (absolute) angles.
8. **How does Position Interpolation extend context?** Linearly rescale
   positions into the trained range (`pos → pos / s`); slightly blurs resolution,
   so a short fine-tune restores quality.
9. **What's NTK-aware scaling and why is it better than linear?** It scales the
   RoPE *base* instead of positions, interpolating low-frequency (coarse) dims
   more and high-frequency (fine) dims less — often works training-free.
10. **Does RoPE add parameters or FLOPs?** No parameters; a small elementwise
    cost (two multiplies + a rotate) on Q/K per layer, with a cached cos/sin
    table.
11. **Why must `head_dim` be even for RoPE?** Dimensions are rotated in 2-D
    pairs; an odd dimension has no partner.
12. **How does RoPE interact with the KV cache (foreshadowing M2)?** You cache
    the *rotated* K (and V); each new query is rotated by its own position, so
    cached keys need no re-rotation — RoPE is KV-cache-friendly.

---

## 1.3 Modern block components (RMSNorm · SwiGLU · QK-norm)

The GPT-2 block is `LayerNorm → MHA → LayerNorm → GELU-MLP`. Modern LLMs
(LLaMA/Qwen/Mistral/Gemma) swap in three components, all now config-driven:

```python
GPTConfig(preset="tiny", norm="rmsnorm", activation="swiglu", qk_norm=True)
```

```bash
python modules/m1_fundamentals/block_components.py            # fast (CI)
python modules/m1_fundamentals/block_components.py --full     # + train compare
```

### RMSNorm (`src/models/norms.py`)

LayerNorm re-centers *and* re-scales: `γ·(x − μ)/√(σ² + ε) + β`. **RMSNorm drops
the mean-centering and the bias**, keeping only the re-scaling:

```
RMSNorm(x) = γ · x / √(mean(x²) + ε)
```

- **Real win:** half the parameters (no `β`) and fewer ops (one reduction, not
  two; no subtraction). Confirmed: `param_reduction = 0.5` in the benchmark.
- **Honest caveat on speed:** our *educational* RMSNorm (unfused Python ops + an
  fp32 cast for stability) is actually **slower wall-clock** than PyTorch's fused
  C++ `nn.LayerNorm` on CPU. RMSNorm's fewer-FLOPs advantage only shows up with a
  fused kernel (Triton/CUDA/`torch.compile`). The benchmark reports this rather
  than pretending otherwise — a good reminder that FLOPs ≠ wall-clock.
- Both norms compute internally in fp32 for stability even under bf16.

### SwiGLU (`src/models/feedforward.py`)

A **gated** MLP replacing the GELU MLP:

```
SwiGLU(x) = W_down( SiLU(W_gate·x) ⊙ (W_up·x) )      SiLU(z) = z·σ(z)
```

The `up` branch is modulated elementwise by a SiLU-activated `gate` branch — the
network learns which features to pass. Three matrices instead of two, so to keep
params comparable we use the **`2/3 · 4d` convention**: hidden dim =
`swiglu_hidden_dim(d_ff) = round(2·d_ff/3)` to a multiple of 8. Benchmark
confirms the match: at `d_model=768, d_ff=3072` GELU and SwiGLU are within
**0.08%** on params (4.722M vs 4.719M).

### QK-norm

An RMSNorm applied to each head's **Q and K** (over `head_dim`) before scoring
(order: `project → qk_norm → RoPE → scores`). Bounds attention-logit magnitude
and stabilizes training at scale / high learning rates (Gemma-2, Chameleon,
Dabra). Cheap: two small norms per attention layer.

### Interview questions

1. **RMSNorm vs LayerNorm — what's dropped and why is it fine?** Mean-centering
   and bias. The re-scaling does the heavy lifting for LM stability; removing the
   mean costs no measurable quality while saving params/FLOPs.
2. **Why compute norms in fp32 even for a bf16 model?** The variance/RMS
   reduction is precision-sensitive; a bf16 reduction can lose significant bits
   and destabilize training. Cast up for the reduction, back down after.
3. **Does fewer FLOPs mean faster? (trap)** No — wall-clock is dominated by
   kernel fusion and memory bandwidth. An unfused RMSNorm can lose to a fused
   LayerNorm; the win needs a fused kernel.
4. **What is a gated activation and why does SwiGLU beat GELU?** A branch that
   multiplicatively gates another (`SiLU(Wg x) ⊙ (Wu x)`). The multiplicative
   interaction adds expressivity; empirically lower loss at equal params.
5. **Explain the 2/3·4d hidden-dim rule.** SwiGLU has 3 weight matrices vs a GELU
   MLP's 2; shrinking hidden to 2/3 keeps total params (and FLOPs) roughly equal
   for an apples-to-apples swap.
6. **What is SiLU/swish?** `x·σ(x)` — a smooth, non-monotonic activation; the
   gate in SwiGLU. (GEGLU uses GELU instead of SiLU — same gating idea.)
7. **Why do LLaMA-style FFNs drop biases?** Negligible quality impact at scale,
   and dropping them simplifies/speeds the layer; norms already handle shift.
8. **What problem does QK-norm solve?** Attention logits can grow large and cause
   instabilities (loss spikes) with big LRs or long training; normalizing Q/K
   bounds the logits.
9. **Where does QK-norm sit relative to RoPE?** Before rotation — normalize the
   raw projected Q/K, then apply RoPE, then score.
10. **Pre-LN vs Post-LN — which does this repo use and why?** Pre-LN (norm
    *before* attention/FFN, inside the residual branch). It keeps a clean
    residual path and is far more stable to train than the original Post-LN.
11. **If you had to pick the single highest-impact GPT-2→modern change, which?**
    Usually RoPE (M1.2) for positional generalization; RMSNorm/SwiGLU are
    efficiency/quality refinements rather than capability unlocks.

---

## 1.4 Attention anatomy (shapes · masks · entropy & sinks)

`attention_deep_dive.py` traces attention from `(B,T,d)` through heads and back
with a shape comment on every line, implements the causal mask variants, and
runs entropy / attention-sink analysis on a real forward pass.

```bash
python modules/m1_fundamentals/attention_deep_dive.py            # fast (CI)
python modules/m1_fundamentals/attention_deep_dive.py --viz      # + attention-map PNG
```

### Shapes, end to end

```
x            (B, T, d)
Q,K,V = xWq,xWk,xWv        (B, T, d)  ->  split heads  ->  (B, H, T, Dh)   Dh=d/H
scores = QKᵀ / √Dh          (B, H, T, T)
scores += mask              (-inf above diagonal for causal)
attn   = softmax(scores)    (B, H, T, T)   rows sum to 1
ctx    = attn · V           (B, H, T, Dh)  ->  merge heads  ->  (B, T, d)
out    = ctx · Wo           (B, T, d)
```

Verified against `F.scaled_dot_product_attention` in `test_attention_anatomy.py`.

### Causal mask variants

- **Additive (−inf)** vs **boolean**: add `-inf` to disallowed scores, or
  `masked_fill` them before softmax — **provably identical** outputs (softmax of
  `-inf` → 0). Additive composes with other biases (ALiBi, sliding window).
- **Sliding-window** (Mistral/Gemma local attention): query `i` sees only keys
  in `(i − w, i]`. Foreshadows M3's hybrid local/global stacking. The window
  bounds KV-cache size to `w`, decoupling it from sequence length.

### Entropy & attention sinks

- **Entropy** (nats) of each query's attention row: `0` = peaked on one token,
  `log T` = uniform. Untrained models sit near `log T` (diffuse); trained models
  develop low-entropy, specialized heads (induction, previous-token, etc.).
- **Attention sink**: the share of attention mass on **token 0**. Trained decoders
  dump "unused" attention onto the first token (often BOS) as a no-op. This is
  why **StreamingLLM keeps the first few tokens** when evicting KV-cache entries
  (M2), and why some models add a learned sink token.

### Interview questions

1. **Why divide scores by √d_k?** Dot products of `d_k`-dim vectors have variance
   ∝ `d_k`; without scaling, softmax saturates into near-one-hot with vanishing
   gradients. `1/√d_k` keeps logits at unit scale.
2. **Additive vs boolean masking — any output difference?** None: softmax of a
   `-inf` logit is 0, same as zeroing the weight. Additive is preferred because
   it composes with other additive biases.
3. **How does the causal mask prevent future leakage, exactly?** It sets
   `scores[i, j>i] = -inf`, so query `i` gets zero weight on future keys — the
   attention matrix is lower-triangular.
4. **Why multiple heads instead of one big head?** Heads attend to different
   subspaces/relations in parallel (syntax, coreference, position); concatenating
   them is more expressive than one head of the same total width.
5. **What's an attention sink and why does it happen?** Softmax must sum to 1, so
   when a head has "nothing to attend to" it parks mass on a stable token
   (usually position 0). It's a pressure-release valve, not meaningful attention.
6. **How do sinks affect KV-cache eviction?** Dropping token 0 tanks quality
   (StreamingLLM finding); keep a few initial "sink" tokens plus a sliding window
   to stream indefinitely with bounded cache.
7. **What does attention entropy tell you?** How focused a head is. Falling
   entropy over training signals heads specializing; a stuck-high-entropy head is
   doing little.
8. **What is an induction head?** A head (often paired with a previous-token
   head) that implements "if `A B` appeared, after a later `A` predict `B`" —
   the mechanistic basis of in-context learning.
9. **Sliding-window attention — what does it cost and save?** Caps attention to
   `O(T·w)` instead of `O(T²)` and bounds KV cache to `w`, at the cost of no
   direct long-range links (recovered by stacking / a few global layers).
10. **Where does the O(n²) in attention come from, and what attacks it?** The
    `T×T` score matrix. KV cache (M2.1) removes recompute; Flash/tiled attention
    (M2.3) removes the memory blow-up; linear/SSM (M3) change the complexity
    class.
11. **Why apply the mask before softmax, not after?** Masking after softmax would
    leave nonzero pre-mask mass and require renormalization; `-inf` before
    softmax cleanly zeros disallowed positions and keeps rows normalized.

---

**M1 complete.** RoPE + RMSNorm + SwiGLU + QK-norm are all config-driven via
`GPTConfig` and become the modern defaults available to every later module.
Benchmarks are committed under `benchmarks/results/m1_*`.
