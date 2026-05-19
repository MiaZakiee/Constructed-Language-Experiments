# LLM Metrics Audit Report

## 1. Mismatch Analysis (Paper vs Computed Values)

We extracted the final average BLEU, chrF, TER, and BERTScore for all 9 models computed by `metrics_aggregation.ipynb` and compared them to Table 1 from the paper.

**Summary**: 0 values were confirmed as exact matches. All values exhibit discrepancies.

### Exact Discrepancies (Paper Value vs. Notebook Computed Value)

| Model | Metric | Paper Table 1 | Notebook Computed (Macro-Average) | Difference |
|-------|--------|---------------|-----------------------------------|------------|
| Claude 3.5 | BLEU | 35.25 | 35.12 | -0.13 |
| Claude 3.5 | chrF | 53.43 | 53.18 | -0.25 |
| Claude 3.5 | TER | 78.31 | 81.81 | +3.50 |
| Claude 3.5 | BERTScore | 0.918 | 0.919 | +0.001 |
| Gemini 2.5 | BLEU | 38.62 | 38.96 | +0.34 |
| Gemini 2.5 | chrF | 52.66 | 53.21 | +0.55 |
| Gemini 2.5 | TER | 67.82 | 70.45 | +2.63 |
| Gemini 2.5 | BERTScore | 0.894 | 0.904 | +0.010 |
| GPT 5 | BLEU | 27.95 | 30.54 | +2.59 |
| GPT 5 | chrF | 44.36 | 46.48 | +2.12 |
| GPT 5 | TER | 87.90 | 91.90 | +4.00 |
| GPT 5 | BERTScore | 0.902 | 0.906 | +0.004 |
| Llama 3 (1B) | BLEU | 3.64 | 4.53 | +0.89 |
| Llama 3 (1B) | chrF | 10.53 | 11.56 | +1.03 |
| Llama 3 (1B) | TER | 119.75 | 129.05 | +9.30 |
| Llama 3 (1B) | BERTScore | 0.796 | 0.801 | +0.005 |
| Llama 3 (8B) | BLEU | 16.72 | 18.15 | +1.43 |
| Llama 3 (8B) | chrF | 36.74 | 38.91 | +2.17 |
| Llama 3 (8B) | TER | 110.11 | 111.03 | +0.92 |
| Llama 3 (8B) | BERTScore | 0.856 | 0.860 | +0.004 |
| Mistral | BLEU | 17.23 | 18.95 | +1.72 |
| Mistral | chrF | 40.32 | 42.26 | +1.94 |
| Mistral | TER | 172.12 | 177.30 | +5.18 |
| Mistral | BERTScore | 0.844 | 0.845 | +0.001 |
| Phi 3 (14B) | BLEU | 19.02 | 20.10 | +1.08 |
| Phi 3 (14B) | chrF | 40.50 | 41.61 | +1.11 |
| Phi 3 (14B) | TER | 167.37 | 166.97 | -0.40 |
| Phi 3 (14B) | BERTScore | 0.850 | 0.852 | +0.002 |
| Phi 3 (8B) | BLEU | 11.64 | 11.27 | -0.37 |
| Phi 3 (8B) | chrF | 33.85 | 33.87 | +0.02 |
| Phi 3 (8B) | TER | 377.82 | 458.80 | +80.98 |
| Phi 3 (8B) | BERTScore | 0.846 | 0.844 | -0.002 |
| Qwen (14B) | BLEU | 27.44 | 29.30 | +1.86 |
| Qwen (14B) | chrF | 49.89 | 50.25 | +0.36 |
| Qwen (14B) | TER | 85.21 | 93.05 | +7.84 |
| Qwen (14B) | BERTScore | 0.872 | 0.873 | +0.001 |

## 2. Aggregation Logic Inconsistencies

1. **Incorrect Macro-Averaging over Categories:** 
   The logic in `metrics_aggregation.ipynb` utilizes `df.groupby(['llm', 'round']).agg('mean')`. This takes an unweighted average of the task categories within a round, completely ignoring the `sample_count` (number of prompts) for each category.
2. **Incorrect Macro-Averaging over Rounds:** 
   The logic then performs `df.groupby('llm').mean()`. This unweighted mean treats each round equally. Because `zero_shot` and `morphological_induction` only have data for Round 1, Round 1 averages 5 categories, while Rounds 2-5 average 3 categories. The final calculation treats all 5 rounds identically, heavily skewing the results against the tasks only present in Round 1.
3. **Local vs Online Model Code Paths:**
   Both local models (Llama, Mistral, Phi, Qwen) and online frontier models (Claude, Gemini, GPT) are processed via the exact same script logic (`CLEANED_DATA_DIR.iterdir()`). No separate code paths exist for the two subsets.

## 3. Recommended Fix
Instead of `df.groupby('llm').mean()`, the script must perform a weighted average at every step by taking `sum(metric * sample_count) / sum(sample_count)` or calculating micro-averages directly over all items.
