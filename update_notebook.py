import json
import pandas as pd
from pathlib import Path

# Fix the notebook
with open('metrics_aggregation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and remove the old "Task 6" cell if it exists
nb['cells'] = [cell for cell in nb['cells'] if not (cell['cell_type'] == 'code' and any('Task 6' in line for line in cell.get('source', [])))]

new_cell = {
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "# Task 6: Export per-task category composite scores for Figures 2 & 3\n",
  "def map_task_category(cat):\n",
  "    cat_lower = cat.lower()\n",
  "    if 'grammatical' in cat_lower or 'gramatical' in cat_lower or 'grammar' in cat_lower:\n",
  "        return 'Grammar'\n",
  "    elif 'morphological' in cat_lower or 'morphology' in cat_lower:\n",
  "        return 'Morphology'\n",
  "    elif 'word_question' in cat_lower or 'vocabulary' in cat_lower:\n",
  "        return 'Vocabulary'\n",
  "    elif 'zero_shot' in cat_lower:\n",
  "        return 'Zero-Shot'\n",
  "    elif 'translation_question' in cat_lower:\n",
  "        return 'Translation'\n",
  "    return cat\n",
  "\n",
  "df['task_category'] = df['category'].apply(map_task_category)\n",
  "\n",
  "task_summary_df = df.groupby(['llm', 'round', 'task_category']).apply(lambda g: pd.Series({\n",
  "    'avg_bleu_score': wavg(g, 'avg_bleu_score', 'sample_count'),\n",
  "    'avg_chrF_score': wavg(g, 'avg_chrF_score', 'sample_count'),\n",
  "    'avg_ter_test': wavg(g, 'avg_ter_test', 'sample_count'),\n",
  "    'avg_bert_score_f1': wavg(g, 'avg_bert_score_f1', 'sample_count'),\n",
  "    'sample_count': g['sample_count'].sum()\n",
  "})).reset_index()\n",
  "\n",
  "task_summary_df['composite_score'] = (\n",
  "    task_summary_df['avg_bleu_score'] + \n",
  "    task_summary_df['avg_chrF_score'] + \n",
  "    (task_summary_df['avg_bert_score_f1'] * 100) - \n",
  "    task_summary_df['avg_ter_test']\n",
  ")\n",
  "\n",
  "task_breakdown_dir = OUTPUT_DIR / 'task_breakdowns'\n",
  "task_breakdown_dir.mkdir(exist_ok=True)\n",
  "\n",
  "for task in ['Vocabulary', 'Grammar', 'Morphology']:\n",
  "    task_df = task_summary_df[task_summary_df['task_category'] == task]\n",
  "    if not task_df.empty:\n",
  "        csv_path = task_breakdown_dir / f\"{task.lower()}_composite_scores.csv\"\n",
  "        task_df[['llm', 'round', 'composite_score']].to_csv(csv_path, index=False)\n",
  "        print(f\"Saved {task} composite scores to {csv_path}\")\n",
  "\n",
  "# Also save Zero-Shot separately just in case\n",
  "zs_df = task_summary_df[task_summary_df['task_category'] == 'Zero-Shot']\n",
  "if not zs_df.empty:\n",
  "    csv_path = task_breakdown_dir / \"zero_shot_composite_scores.csv\"\n",
  "    zs_df[['llm', 'round', 'composite_score']].to_csv(csv_path, index=False)\n",
  "    print(f\"Saved Zero-Shot composite scores to {csv_path}\")"
 ]
}

nb['cells'].append(new_cell)

with open('metrics_aggregation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Notebook updated successfully!')
