
**Cold Start (Fine-Tune)**

- **What**: Fine-tuning of DeepSeek-V3-Base
- **Why**: Give the model basic reasoning capabilities / patterns, stable RL foundation, output fmt
- **Data**: Thousands CoT examples

**RL Stage 1**

- What: GRPO reinforcement learning
    - https://arxiv.org/abs/2402.03300
    - 64 samples per training example
    - Score each one w/ a rule-based reward (e.g., correct/incorrect for math, coding)
    - Compare each sample to the mean of all samples
    - For samples with high (low) normalized reward above (below) the group mean:
        - Increase (drop) probability of the model generating *all* the tokens in that sequence.
        - Each token in the sequence get a positive (negative) gradient update.
        - *Let’s make all the choices that led to it more (or less) likely in the future.*
- Why: "Discover" good reasoning patterns, makes the model very strong at reasoning
    - Lost some general capabilities
    - Had potential language mixing issues.
- Data: ~144K CoT format GSM8K and MATH questions on reasoning-intensive tasks

**Rejection Sampling**

- What: Generate new training data by filtering results from RL stage 1
- Why: Results in 600k reasoning traces
- Data: 600k + 200k non-reasoning (writing, factual QA, etc) DeepSeek-V3's SFT dataset

**Fine-Tune** 

- What: Fine-tuning of RL stage 1 model
- Why: “Bake” reasoning patterns in, restore general capabilities
- Data: Data from rejection sampling, ~800k samples

**RL stage 2** 

- What: GRPO reinforcement learning
    - Rule-based reward (stage 1) for reasoning
    - Reward models for helpfulness and harmlessness on general data
- Why: Optimizing for both reasoning + general capabilities
- Data: Mix of reasoning data and general data

### Results

- On par with o1
- **Mathematical Reasoning**: AIME 2024, MATH-500, GPQA Diamond
- **Coding**: Codeforces, SWE-bench Verified

- Fine-tune on quality reasoning tracing
- 14b similar to o1-mini

---

### 1. Input Embedding

**What It Is**

Input embedding transforms raw input tokens (e.g., words or subwords) into
dense, continuous vector representations in a high-dimensional space. This
converts discrete symbols into a format suitable for neural network processing.

**How It Works**

*   **Tokenization:** Text is split into tokens (e.g., "cat" or "##ing") using a
    vocabulary.
*   **Embedding Layer:** Each token is mapped to a vector using a learned
    lookup table. For a vocabulary of size $V$ and embedding dimension $d$,
    the embedding matrix $E \in \mathbb{R}^{V \times d}$ assigns a vector
    $e_i \in \mathbb{R}^d$ to token $i$.
*   **Mathematical Representation:** If $x_t$ is the token index at position
    $t$, the embedding is:

    $$e_t = E[x_t]$$
*   **Positional Encoding:** Since transformers lack recurrence, positional
    information is added to embeddings using sine and cosine functions:

    $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad
    PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

    where $pos$ is the token position and $i$ is the dimension index. This
    allows the model to understand sequence order.

**Interesting Insights**

*   **Dimensionality:** Typical $d$ ranges from 256 to 1024, balancing
    expressiveness and computational cost.
*   **Learnability:** Embeddings evolve during training, capturing semantic
    relationships (e.g., "king" and "queen" are closer in vector space).
*   **Why It Matters:** This step bridges human language and machine learning,
    enabling the model to "understand" context from the outset.

### 2. Self-Attention

**What It Is**

Self-attention is a mechanism that allows the model to weigh the importance of
different tokens in a sequence when processing each token, capturing
contextual relationships dynamically.

**How It Works**

*   **Query, Key, Value (QKV):** For each token, three vectors are computed
    from its embedding: query $q$, key $k$, and value $v$, derived via
    linear transformations:

    $$q = W_Q e_t, \quad k = W_K e_t, \quad v = W_V e_t$$

    where $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$ are weight matrices,
    and $d_k$ is the attention dimension (often $d_k = d/h$, with $h$
    heads).
*   **Attention Score:** The compatibility between $q_i$ and all $k_j$ (for
    tokens $j$) is calculated using dot products, scaled by
    $\sqrt{d_k}$ to prevent large values:

    $$\text{Score}(i, j) = \frac{q_i \cdot k_j}{\sqrt{d_k}}$$
*   **Softmax Normalization:** Converts scores into attention weights:

    $$\alpha_{ij} = \frac{\exp(\text{Score}(i, j))}{\sum_{m}
    \exp(\text{Score}(i, m))}$$
*   **Weighted Sum:** The output for token $i$ is a weighted combination of
    all values:

    $$\text{Attention}(i) = \sum_{j} \alpha_{ij} v_j$$
*   **Multi-Head Attention:** Parallel attention heads $h$ process different
    subspaces, concatenating results:

    $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots,
    \text{head}_h) W_O$$

    where $\text{head}_i = \text{Attention}(Q W_Q^i, K W_K^i, V W_V^i)$ and
    $W_O \in \mathbb{R}^{hd_k \times d}$.

**Interesting Insights**

*   **Parallelism:** Unlike RNNs, self-attention processes all tokens
    simultaneously, enabling efficient training on GPUs.
*   **Context Awareness:** A token like "it" can attend to "cat" or "ran" based
    on context (e.g., "The cat ran and it stopped"), showcasing dynamic
    weighting.
*   **Scalability:** Attention complexity is $O(n^2)$, where $n$ is sequence
    length, driving innovations like sparse attention to handle longer
    contexts.

### 3. Transformer Architecture

**What It Is**

The Transformer, introduced in "Attention is All You Need" (Vaswani et al.,
2017), is an encoder-decoder architecture relying entirely on self-attention
and feed-forward networks, replacing recurrent layers.

**How It Works**

*   **Encoder:** Comprises $N$ identical layers (e.g., 6), each with:
    *   **Multi-Head Self-Attention:** Captures intra-sequence relationships.
    *   **Feed-Forward Network (FFN):** Applies a position-wise
        transformation:

        $$\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2$$

        where $W_1, W_2$ are weight matrices.
    *   **Layer Normalization and Residual Connections:** Stabilizes training
        with:

        $$\text{LayerOutput} = \text{LayerNorm}(x + \text{SubLayer}(x))$$

*   **Decoder:** Similar to the encoder but includes masked self-attention (to
    prevent attending to future tokens) and encoder-decoder attention for
    generation.
*   **Training Objective:** Minimizes cross-entropy loss over predicted token
    probabilities, optimized with techniques like Adam.

**Interesting Insights**

*   **Versatility:** Used in NLP (e.g., BERT, GPT) and beyond (e.g., vision
    transformers), showcasing adaptability.
*   **Efficiency:** The absence of recurrence allows parallel processing,
    reducing training time from days (RNNs) to hours on modern hardware.
*   **General Idea:** Transformers "think" by dynamically weighting
    relationships across the entire input, mimicking human contextual
    understanding, which is key to LLMs' reasoning prowess.

### Why These Concepts Matter

*   **Input Embedding** lays the groundwork by encoding meaning and position,
    enabling the model to interpret language.
*   **Self-Attention** provides the flexibility to focus on relevant tokens,
    capturing long-range dependencies critical for reasoning tasks.
*   **Transformer Architecture** integrates these into a scalable,
    parallelizable framework, powering state-of-the-art models like
    DeepSeek-R1 by efficiently handling complex language patterns.

This foundation equips readers with a solid grasp of how transformers process
and generate language, setting the stage for advanced applications in AI.


# DeepSeek-R1 Training Process: Advancing Reasoning in LLMs

The **DeepSeek-R1 framework** revolutionizes reasoning capabilities in large language models (LLMs) through a **multi-phase training strategy**, integrating reinforcement learning (RL) and supervised fine-tuning (SFT). Built upon DeepSeek-V3-Base, it produces DeepSeek-R1-Zero (pure RL) and DeepSeek-R1 (hybrid refinement), culminating in distilled Dilled-R1 models (1.5B to 70B parameters). This process rivals OpenAI's o1-1217, achieving **79.8% pass@1 on AIME 2024**, by systematically enhancing reasoning, generalizability, and ethical alignment.

## 1. Foundation: DeepSeek-V3-Base

* **Concept:** DeepSeek-V3-Base is a pre-trained transformer model, optimized on a vast text corpus, providing a broad linguistic foundation.
* **Why:** Starting with a pre-trained model leverages transfer learning, reducing training costs and enabling RL to focus on reasoning rather than basic language acquisition. This exploits the model's existing probability distribution over tokens, refined during pre-training.
* **Math:** Pre-training minimizes cross-entropy loss to predict the next token:
    $$
    L_{\text{pre-train}} = -\frac{1}{N} \sum_{i=1}^N \log p(y_i | x_{<i}; \theta)
    $$
    where $N$ is the sequence length, $y_i$ is the target token, $x_{<i}$ is the context, and $\theta$ are the model parameters. This establishes a strong initial policy $\pi_{\text{base}}(a|s; \theta)$.
* **Purpose:** Provides a stable, general-purpose starting point for specialized reasoning development via RL.

---

## 2. DeepSeek-R1-Zero: Pure RL with Group Relative Policy Optimization (GRPO)

* **Concept:** R1-Zero trains solely with RL, using **GRPO**, a novel algorithm that eliminates the critic network by leveraging group-based reward comparisons.
* **Why:** This tests the hypothesis that RL incentives alone can evoke emergent reasoning, bypassing the need for extensive labeled CoT data. It explores the model's latent reasoning potential through self-evolution, though it risks instability without initial guidance.
* **Math & Mechanics:**
    * **Sampling:** For each prompt, GRPO generates a group of $K$ samples (e.g., $K=20$) from the policy $\pi_\theta(a|s)$, representing diverse reasoning paths.
    * **Reward Model:** Combines accuracy ($R_{\text{acc}}$) and formatting ($R_{\text{format}}$) rewards:
        $$
        R(s, a) = w_1 R_{\text{acc}}(s, a) + w_2 R_{\text{format}}(s, a)
        $$
        where $R_{\text{acc}} \in \{0, 1\}$ for correctness (e.g., math solution validation), and $R_{\text{format}} = -w \cdot \text{error\_count}$ penalizes structural deviations (e.g., missing `<think>` tags).
    * **Objective:** Maximizes expected cumulative reward:
        $$
        J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^T \gamma^t R(s_t, a_t) \right]
        $$
        with a trust region constraint via KL-divergence:
        $$
        D_{\text{KL}}(\pi_\theta || \pi_{\text{ref}}) < \delta
        $$
        where $\gamma \in (0.9, 0.99)$ discounts future rewards, and $\delta$ ensures stability. The advantage function uses a group baseline:
        $$
        A_t = R_t - \frac{1}{K} \sum_{k=1}^K R_k
        $$
        encouraging the policy to favor high-performing samples within the group.
* **Emergence:** After ~10K RL iterations, R1-Zero exhibits "**aha moments**" (self-correction when $A_t > 0$), increasing CoT length and boosting AIME 2024 pass@1 from 15.6% to 71.0%, nearing OpenAI-o1-0912. However, this introduces readability issues (e.g., repetition) and language mixing due to unguided exploration.
* **Purpose:** Validates RL's capacity to unlock reasoning autonomously, setting a baseline for structured refinement.

---

## 3. DeepSeek-R1: Multi-Stage Refinement Pipeline

DeepSeek-R1 addresses R1-Zero's limitations with a hybrid approach, balancing reasoning depth and general skills.

### Cold-Start Supervised Fine-Tuning (SFT)

* **Concept:** Fine-tunes V3-Base on a small, high-quality CoT dataset (e.g., 10K examples) with structured formats like `|special_token|<reasoning>|special_token|<answer>|`.
* **Why:** Initializes the model with human-verified reasoning patterns, mitigating RL instability and improving output coherence. This reduces the search space for RL, focusing it on optimization rather than exploration.
* **Math:** Optimizes cross-entropy loss over the dataset:
    $$
    L_{\text{SFT}} = -\frac{1}{M} \sum_{j=1}^M \log p(y_j | x_j; \theta)
    $$
    where $M$ is the number of examples, guiding $\theta$ toward structured reasoning.
* **Purpose:** Establishes a readable, consistent reasoning foundation for RL.

### Reasoning-Oriented RL with Enhanced GRPO

* **Concept:** Applies GRPO to deepen reasoning, adding a **language consistency reward** ($R_{\text{lang}}$).
* **Why:** Enhances reasoning precision on tasks like math and coding while addressing language mixing, though it trades off minor accuracy (1-2% drop) for coherence.
* **Math:** Updates the reward function:
    $$
    R(s, a) = w_1 R_{\text{acc}}(s, a) + w_2 R_{\text{format}}(s, a) + w_3 R_{\text{lang}}(s, a)
    $$
    where $R_{\text{lang}} = \frac{\text{number of target language tokens}}{\text{total number of tokens}}$ encourages monolingual output. GRPO optimizes:
    $$
    \nabla_\theta J(\theta) \propto \mathbb{E} \left[ A_t \nabla_\theta \log \pi_\theta(a_t | s_t) \right]
    $$
    with $A_t$ adjusted by the group baseline, achieving **79.8% AIME pass@1**, equaling OpenAI-o1-1217.
* **Purpose:** Refines reasoning depth and output quality, aligning with human expectations.

### General SFT with Rejection Sampling

* **Concept:** Generates 800k samples from the RL checkpoint, filters top performers via rejection sampling ($R > \tau$), and fine-tunes on a 200k mix of reasoning and general data (writing, QA).
* **Why:** Balances reasoning with versatility, ensuring helpfulness and harmlessness. Rejection sampling amplifies high-reward behaviors, while mixed data prevents overfitting to reasoning tasks.
* **Math:** Employs a weighted loss:
    $$
    L = \alpha L_{\text{reasoning}}(y_r, \hat{y}_r) + (1 - \alpha) L_{\text{general}}(y_g, \hat{y}_g)
    $$
    where $\alpha \in [0, 1]$ tunes the trade-off, and a reward model scores general outputs for nuanced alignment.
* **Purpose:** Produces a holistic DeepSeek-R1, excelling across domains while adhering to ethical standards.

---

## 4. Knowledge Distillation: Dilled-R1 Models

* **Concept:** Distills DeepSeek-R1 into smaller models (e.g., Qwen-14B) using **soft targets** from the teacher model.
* **Why:** Reduces computational overhead for deployment, preserving reasoning prowess. Pure RL on small models (e.g., Qwen-32B) underperforms due to limited capacity, making distillation essential.
* **Math:** Minimizes a distillation loss:
    $$
    L_{\text{distill}} = \beta L_{\text{CE}}(y, \hat{y}) + (1 - \beta) \text{KL}(p_{\text{teacher}} || p_{\text{student}})
    $$
    where $\beta$ balances hard labels ($L_{\text{CE}}$) and soft target alignment (KL-divergence), and $p_{\text{teacher}}$ is DeepSeek-R1's output distribution. A 14B model retains **69.7% AIME pass@1**.
* **Purpose:** Enables scalable, efficient reasoning-capable LLMs for widespread use.

---

## Insights & Impact

DeepSeek-R1 showcases RL's transformative potential, achieving **97.3% on MATH-500** and **71.5% on GPQA Diamond**. The "**aha moment**" reflects positive advantage-driven self-correction, while distillation democratizes access. Challenges like language mixing suggest future integration of multilingual reward terms, enhancing global applicability.


## DeepSeek-R1 Reasoning Capability: Solving a Radical Equation

This document summarizes DeepSeek-R1's step-by-step solution to the problem: "If $a > 1$, then the sum of the real solutions of $\sqrt{a - \sqrt{a + x}} = x$ is equal to...". The solution demonstrates the model's Chain-of-Thought (CoT) approach and self-correction, enhanced by reinforcement learning (RL).

### Problem Statement

Solve for the sum of real solutions of the equation:

$$\sqrt{a - \sqrt{a + x}} = x$$

where $a > 1$.

### Solution Steps

1.  **Initial Squaring:** Squaring both sides to eliminate the outer square root:

    $$a - \sqrt{a + x} = x^2$$
2.  **Attempted Second Squaring (Initial):** Squaring again leads to:

    $$a + x = (a - x^2)^2$$
3.  **Expansion and Quartic Equation (Initial):** Expanding the square:

    $$a + x = a^2 - 2ax^2 + x^4$$

    Rearranging into a quartic equation:

    $$x^4 - 2ax^2 - x + (a^2 - a) = 0$$

4.  **"Aha Moment" (Self-Correction):** Recognizing that the quartic equation is overly complex and may introduce extraneous solutions, the model reevaluates the approach.
5.  **Isolation of the Radical (Corrected):** Returning to the equation after the first squaring and isolating the inner radical:

    $$\sqrt{a + x} = a - x^2$$

    This requires $a - x^2 \geq 0$ and $a + x \geq 0$.
6.  **Squaring Again (Corrected):** Squaring both sides:

    $$a + x = (a - x^2)^2$$

    Which leads back to the quartic:

    $$x^4 - 2ax^2 - x + (a^2 - a) = 0$$

7.  **Solving the Quartic and Sum of Solutions:** For a general quartic equation $x^4 + bx^3 + cx^2 + dx + e = 0$, the sum of the roots is $-b$ (Vieta's formulas). In this case, $b = 0$, so the sum of roots is:

    $$\text{Sum of roots} = -0 = 0$$

8.  **Domain and Reality Check:** The solution must satisfy $\sqrt{a - \sqrt{a + x}} = x$, requiring $a - \sqrt{a + x} \geq 0$ and $x \geq 0$.

### Key Takeaways

*   **Self-Correction:** DeepSeek-R1 demonstrates the ability to detect and correct algebraic missteps through RL.
*   **Mathematical Rigor:** The process involves iterative squaring, polynomial manipulation, and consideration of domain constraints.
*   **Reasoning Depth:** The step-by-step reevaluation mimics human problem-solving.

The sum of real solutions to the equation is likely 0 after filtering for valid solutions.
