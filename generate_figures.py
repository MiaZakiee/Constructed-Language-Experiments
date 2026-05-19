import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")

os.makedirs('figures', exist_ok=True)

MODEL_LABELS = {
    'claude3.5': 'Claude 3.5',
    'gemini2.5': 'Gemini 2.5',
    'gpt5': 'GPT-5',
    'llama3_1b': 'Llama 3 (1B)',
    'llama3_8b': 'Llama 3 (8B)',
    'mistral': 'Mistral',
    'phi3_14b': 'Phi3 (14B)',
    'phi3_8b': 'Phi3 (8B)',
    'qwen_14b': 'Qwen (14B)'
}

FRONTIER = ['claude3.5', 'gemini2.5', 'gpt5']
MID = ['qwen_14b', 'phi3_14b', 'llama3_8b', 'mistral']
SMALL = ['phi3_8b', 'llama3_1b']

# Generate distinct colors for lines
colors = sns.color_palette("husl", 9)
MODEL_COLORS = {model: colors[i] for i, model in enumerate(MODEL_LABELS.keys())}


def plot_line_chart(csv_path, title, output_path):
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(12, 7))
    
    for model, group in df.groupby('llm'):
        if model in FRONTIER:
            linestyle = '-'
            linewidth = 2.5
        elif model in MID:
            linestyle = '--'
            linewidth = 1.5
        else:
            linestyle = ':'
            linewidth = 1.5
            
        plt.plot(
            group['round'], 
            group['composite_score'], 
            label=MODEL_LABELS.get(model, model),
            linestyle=linestyle,
            linewidth=linewidth,
            color=MODEL_COLORS[model]
        )
        
    plt.axhline(y=0, color='red', linestyle='--', label='Generation Collapse Threshold')
    
    plt.xlabel('Prompt Complexity Level (Round)', fontsize=12)
    plt.ylabel('Composite Score', fontsize=12)
    plt.title(title, fontsize=14, pad=20)
    
    plt.xticks(range(1, 6))
    
    # Place legend outside
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_bar_chart(csv_path, title, subtitle, output_path):
    df = pd.read_csv(csv_path)
    
    # Sort by composite score
    df_sorted = df.sort_values(by='composite_score', ascending=True)
    
    # Apply clean labels
    df_sorted['label'] = df_sorted['llm'].map(MODEL_LABELS).fillna(df_sorted['llm'])
    
    plt.figure(figsize=(10, 6))
    
    # Generate colors based on value
    bar_colors = ['green' if x > 0 else 'red' for x in df_sorted['composite_score']]
    
    bars = plt.barh(df_sorted['label'], df_sorted['composite_score'], color=bar_colors)
    
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Collapse Threshold')
    
    plt.xlabel('Composite Score', fontsize=12)
    plt.title(title, fontsize=14, pad=25)
    plt.suptitle(subtitle, fontsize=10, y=0.92, alpha=0.7)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 5 if width > 0 else width - 5
        ha = 'left' if width > 0 else 'right'
        plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                 f'{round(width)}', 
                 ha=ha, va='center', fontsize=10)
        
    # Extend xlim to fit labels if necessary
    xmin, xmax = plt.xlim()
    padding = (xmax - xmin) * 0.1
    plt.xlim(xmin - padding, xmax + padding)
                 
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# Plot Figure 2
plot_line_chart(
    'aggregated_metrics/task_breakdowns/vocabulary_composite_scores.csv',
    'Vocabulary Task: Composite Score by Prompt Complexity Level',
    'figures/vocabulary_composite_by_round.png'
)

# Plot Figure 3
plot_line_chart(
    'aggregated_metrics/task_breakdowns/grammar_composite_scores.csv',
    'Grammar Task: Composite Score by Prompt Complexity Level',
    'figures/grammar_composite_by_round.png'
)

# Plot Figure 4
plot_bar_chart(
    'aggregated_metrics/task_breakdowns/morphology_composite_scores.csv',
    'Morphological Induction: Single-Round Composite Score by Model',
    'Morphology tested in Round 1 only; scores below 0 indicate generation collapse',
    'figures/morphology_composite_bar.png'
)

# Plot Figure 5
plot_bar_chart(
    'aggregated_metrics/task_breakdowns/zero_shot_composite_scores.csv',
    'Zero-Shot Baseline: Composite Score by Model (No Rules Provided)',
    'Near-zero or negative scores confirm models had no prior knowledge of Dasopya',
    'figures/zero_shot_baseline_bar.png'
)

print("Figures generated successfully.")
