"""
Length x Sycophancy Analysis on AITA dataset (Cheng et al. 2026)

Analysis modes:
    'human'    → sycophancy label from human ground truth only (is_asshole + binary_base)
    'llm'      → sycophancy label from LLM judge only (openended_baserated)
    'agree'    → only cases where human and LLM judge agree
    'disagree' → only cases where human and LLM judge disagree
"""

import os
import sys
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats

# FIXME: Need to download file from Cheng et al. 2026 social sycophancy paper
DATA_PATH  = 'AITA_endorsement_results.csv'
OUT_DIR    = './results'
USE_MEDIAN = False
ALL_MODES  = ['human', 'llm', 'agree', 'disagree']

os.makedirs(OUT_DIR, exist_ok=True)
big = pd.read_csv(DATA_PATH, low_memory=False)
models = ['Llama-8B', 'Llama-17B', 'Llama-70B', 'Claude', 'Gemini', 'gpt-4o', 'Mistral-7B', 'Mistral-24B']
results_log = []

for MODE in ALL_MODES:
        # redirect stdout to capture print output
        captured = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured
        print(f"\n{'#'*60}")
        print(f"# MODE: {MODE}")
        print(f"{'#'*60}\n")

        NEEDS_HUMAN = MODE in ('human', 'agree', 'disagree')
        NEEDS_RATED = MODE in ('llm', 'agree', 'disagree')
        ratios = []
        chunks = []
        for m in models:
                bin_col  = f"{m}_binary_base"
                text_col = f"{m}_openended_base"
                rate_col = f"{m}_openended_baserated"

                needed = [bin_col, text_col]
                if NEEDS_HUMAN: needed = ['is_asshole'] + needed
                if NEEDS_RATED: needed = needed + [rate_col]

                if not all(c in big.columns for c in needed):
                        print(f"Skipping {m}: missing columns")
                        continue
                df = big[needed].copy()

                if NEEDS_HUMAN:
                        df = df[big['is_asshole'] == 1]
                df = df.dropna(subset=needed)
                df = df[df[bin_col].str.contains("YTA|NTA", na=False)]

                # define labels per mode
                is_nta = df[bin_col].str.contains("NTA", case=False)

                if NEEDS_RATED:
                        is_rated = df[rate_col].astype(float) == 1.0

                if MODE == 'human':
                        syco     = is_nta
                        non_syco = ~is_nta

                elif MODE == 'llm':
                        syco     = is_rated
                        non_syco = ~is_rated

                elif MODE == 'agree':
                        syco     = is_nta  &  is_rated
                        non_syco = ~is_nta & ~is_rated

                elif MODE == 'disagree':
                        syco     = is_nta  & ~is_rated
                        non_syco = ~is_nta &  is_rated
                df = df[syco | non_syco].copy()
                if len(df) == 0:
                        print(f"Skipping {m}: no rows after filtering")
                        continue

                df['sycophantic']     = syco[df.index]
                df['response_length'] = df[text_col].str.len()
                df['model']           = m
                df = df.rename(columns={text_col: 'response', bin_col: 'binary_verdict'})

                keep_cols = ['model', 'binary_verdict', 'sycophantic', 'response_length', 'response']
                if NEEDS_HUMAN: keep_cols.insert(1, 'is_asshole')
                if NEEDS_RATED: keep_cols.insert(-1, rate_col)
                df = df[keep_cols]
                chunks.append(df)

                stat_fn   = 'median' if USE_MEDIAN else 'mean'
                ratio_val = (df[df['sycophantic'] == False]['response_length'].agg(stat_fn) /
                                         df[df['sycophantic'] == True]['response_length'].agg(stat_fn))
                ratios.append(ratio_val)

        # combined violin plot (3 cols × 3 rows)
        stat_fn = 'median' if USE_MEDIAN else 'mean'
        fig, axes = plt.subplots(3, 3, figsize=(14, 14))
        axes_flat = axes.flatten()

        for ax, (m, df) in zip(axes_flat, [(c['model'].iloc[0], c) for c in chunks]):
                sns.violinplot(x='sycophantic', y='response_length', data=df, inner='box', ax=ax)
                for val in [False, True]:
                        group  = df[df['sycophantic'] == val]['response_length']
                        center = group.median() if USE_MEDIAN else group.mean()
                        boots  = [group.sample(frac=1, replace=True).agg(stat_fn) for _ in range(1000)]
                        ci_low  = pd.Series(boots).quantile(0.025)
                        ci_high = pd.Series(boots).quantile(0.975)
                        x_pos   = 1 if val else 0
                        ax.errorbar(x_pos + 0.1, center,
                                                yerr=[[center - ci_low], [ci_high - center]],
                                                fmt='x', color='k', markersize=6, capsize=6,
                                                linewidth=2, zorder=5,
                                                label=f'{"Median" if USE_MEDIAN else "Mean"} with 95% CI' if val else None)

                ax.set_title(f'[{MODE}] {m}')
                ax.set_xlabel('Sycophantic')
                ax.set_ylabel('Response Length (characters)')
                ax.set_xticks([0, 1])
                ax.set_xticklabels(['Non-sycophantic', 'Sycophantic'])
                ax.set_xlim(-0.5, 1.5)
                ax.legend()

        plt.tight_layout()
        for ax in axes_flat[len(chunks):]:
                ax.set_visible(False)
        plt.tight_layout()

        # save fig
        plot_base = os.path.join(OUT_DIR, f'syc_length_{MODE}')
        fig.savefig(f'{plot_base}.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(f'{plot_base}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        # aggregate results
        result = pd.concat(chunks, ignore_index=True)
        print(f"\n{'='*60}")
        print(f"MODE: {MODE}  |  N responses: {len(result)}")
        print(f"{'='*60}\n")

        fit = smf.mixedlm("response_length ~ sycophantic", result, groups=result["model"]).fit()
        print(fit.summary())
        print(f"\nPer-model ratios (non-syco / syco) [{stat_fn}]:", ratios)
        print(f"Median of ratios: {np.median(ratios):.4f}")
        print(f"\nPer-model distribution tests:")
        for m in models:
                mdf = result[result['model'] == m]
                if mdf.empty:
                        continue
                syco_len     = mdf[mdf['sycophantic'] == True]['response_length']
                non_syco_len = mdf[mdf['sycophantic'] == False]['response_length']
                ks_stat, ks_p = stats.ks_2samp(syco_len, non_syco_len)
                w_dist        = stats.wasserstein_distance(syco_len, non_syco_len)
                print(f"  {m}: KS={ks_stat:.3f} (p={ks_p:.3f}), Wasserstein={w_dist:.1f}")

        result['length_resid'] = result.groupby('model')['response_length'].transform(
                lambda x: (x - x.mean()) / x.std()
        )
        syco_resid     = result[result['sycophantic']]['length_resid']
        non_syco_resid = result[~result['sycophantic']]['length_resid']
        pooled_ks      = stats.ks_2samp(syco_resid, non_syco_resid)
        print(f"\nPooled KS on z-scored residuals: "
                    f"KS={pooled_ks.statistic:.4f}, p={pooled_ks.pvalue:.4f}")

        sys.stdout = original_stdout
        mode_output = captured.getvalue()
        results_log.append(mode_output)
        print(mode_output, end='')   # also print to terminal

# write output
results_path = os.path.join(OUT_DIR, 'syc_length_results.txt')
with open(results_path, 'w') as f:
        f.write(''.join(results_log))

print(f"\nResults saved to {results_path}")
print(f"Plots saved to {OUT_DIR}/syc_length_{{mode}}.pdf / .png")




# Within-prompt across-models analysis
print(f"\n{'#'*60}")
print(f"# WITHIN-PROMPT ACROSS-MODELS ANALYSIS")
print(f"{'#'*60}\n")

wp_chunks = []

for m in models:
    bin_col  = f"{m}_binary_base"
    text_col = f"{m}_openended_base"
    needed   = ['is_asshole', bin_col, text_col]

    if not all(c in big.columns for c in needed):
        print(f"Skipping {m}: missing columns")
        continue

    df = big[needed].copy()
    df = df[big['is_asshole'] == 1]
    df = df.dropna(subset=needed)
    df = df[df[bin_col].str.contains("YTA|NTA", na=False)]

    is_nta = df[bin_col].str.contains("NTA", case=False)

    df = df.copy()
    df['sycophantic']     = is_nta
    df['response_length'] = df[text_col].str.len()
    df['model']           = m
    df['prompt_id']       = df.index   # original row index = prompt identity
    df = df.rename(columns={text_col: 'response', bin_col: 'binary_verdict'})
    df = df[['prompt_id', 'model', 'is_asshole', 'binary_verdict',
             'sycophantic', 'response_length', 'response']]
    wp_chunks.append(df)

wp_result = pd.concat(wp_chunks, ignore_index=True)

# keep only prompts where at least one model is sycophantic AND
# at least one model is non-sycophantic (i.e., variation exists)
prompt_syco_counts = wp_result.groupby('prompt_id')['sycophantic'].agg(
    has_syco  = lambda x: x.any(),
    has_non   = lambda x: (~x).any()
)
varied_prompts = prompt_syco_counts[
    prompt_syco_counts['has_syco'] & prompt_syco_counts['has_non']
].index

wp_varied = wp_result[wp_result['prompt_id'].isin(varied_prompts)].copy()

print(f"Prompts with cross-model variation: {len(varied_prompts)}")
print(f"Total responses in within-prompt analysis: {len(wp_varied)}")
print(f"Sycophantic: {wp_varied['sycophantic'].sum()}, "
      f"Non-sycophantic: {(~wp_varied['sycophantic']).sum()}\n")

# mixed effects: prompt as random effect (absorbs prompt difficulty/content)
# this is the key difference from Step 1 where model was the random effect
wp_fit = smf.mixedlm(
    "response_length ~ sycophantic",
    wp_varied,
    groups=wp_varied["prompt_id"]
).fit()
print(wp_fit.summary())

# per-prompt KS on z-scored residuals (complementary pooled test)
wp_varied['length_resid'] = wp_varied.groupby('prompt_id')['response_length'].transform(
    lambda x: (x - x.mean()) / x.std()
)
# drop prompts where std=0 (all models gave same length — rare but possible)
wp_varied = wp_varied.dropna(subset=['length_resid'])

wp_syco_resid     = wp_varied[wp_varied['sycophantic']]['length_resid']
wp_non_syco_resid = wp_varied[~wp_varied['sycophantic']]['length_resid']
wp_pooled_ks      = stats.ks_2samp(wp_syco_resid, wp_non_syco_resid)
print(f"\nPooled KS on z-scored within-prompt residuals: "
      f"KS={wp_pooled_ks.statistic:.4f}, p={wp_pooled_ks.pvalue:.4f}")

fig, axes = plt.subplots(3, 3, figsize=(14, 14))
axes_flat = axes.flatten()

for ax, m in zip(axes_flat, models):
    mdf = wp_varied[wp_varied['model'] == m]
    if mdf.empty:
        ax.set_visible(False)
        continue

    sns.violinplot(x='sycophantic', y='response_length', data=mdf, inner='box', ax=ax)

    for val in [False, True]:
        group  = mdf[mdf['sycophantic'] == val]['response_length']
        center = group.mean()
        boots  = [group.sample(frac=1, replace=True).mean() for _ in range(1000)]
        ci_low  = pd.Series(boots).quantile(0.025)
        ci_high = pd.Series(boots).quantile(0.975)
        x_pos   = 1 if val else 0
        ax.errorbar(x_pos + 0.1, center,
                    yerr=[[center - ci_low], [ci_high - center]],
                    fmt='x', color='k', markersize=6, capsize=6,
                    linewidth=2, zorder=5,
                    label='Mean with 95% CI' if val else None)

    n_prompts = mdf['prompt_id'].nunique()
    ax.set_title(f'[within-prompt] {m}\n(n={n_prompts} prompts)')
    ax.set_xlabel('Sycophantic')
    ax.set_ylabel('Response Length (characters)')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Non-sycophantic', 'Sycophantic'])
    ax.set_xlim(-0.5, 1.5)
    ax.legend()

for ax in axes_flat[len(models):]:
    ax.set_visible(False)

plt.tight_layout()
plot_base = os.path.join(OUT_DIR, 'syc_length_within_prompt')
fig.savefig(f'{plot_base}.pdf', dpi=300, bbox_inches='tight')
fig.savefig(f'{plot_base}.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# append to results file
with open(results_path, 'a') as f:
    f.write(f"\n{'#'*60}\n# WITHIN-PROMPT ACROSS-MODELS ANALYSIS\n{'#'*60}\n")
    f.write(f"Prompts with variation: {len(varied_prompts)}\n")
    f.write(f"Total responses: {len(wp_varied)}\n")
    f.write(wp_fit.summary().as_text())
    f.write(f"\nPooled KS on z-scored within-prompt residuals: "
            f"KS={wp_pooled_ks.statistic:.4f}, p={wp_pooled_ks.pvalue:.4f}\n")

print(f"\nWithin-prompt results appended to {results_path}")