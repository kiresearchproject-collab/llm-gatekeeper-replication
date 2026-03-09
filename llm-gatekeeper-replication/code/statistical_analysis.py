"""
STATISTICAL ANALYSIS - LLMs as the Gatekeeper
==============================================
This script performs all statistical analyses reported in the paper
"Decision Gatekeepers: How Hedonic vs. Utilitarian Products Shape
Differential Persuasion Effects in LLM-Mediated Recommendations"

USAGE:
    python statistical_analysis.py

REQUIREMENTS:
    pip install pandas numpy scipy statsmodels
"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import warnings
warnings.filterwarnings('ignore')

FILE_PATH = "../data/llm_gatekeeper_dataset.csv"

def compute_eta_squared(anova_table, effect_name):
    """Compute eta-squared: SS_effect / (SS_effect + SS_residual)."""
    ss_effect = anova_table.loc[effect_name, 'sum_sq']
    ss_resid = anova_table.loc['Residual', 'sum_sq']
    return ss_effect / (ss_effect + ss_resid)


def load_data(file_path):
    """Load and prepare the experimental dataset."""
    df = pd.read_csv(file_path)
    for col in df.select_dtypes(include=['string']).columns:
        df[col] = df[col].astype(object)
    df['model_clean'] = df['model_name'].str.replace('openai/', '').str.replace('moonshotai/', '')
    print(f"Loaded {len(df):,} observations")
    return df


def section_4_1(df):
    """Section 4.1: Sample Characteristics."""
    print("\n" + "=" * 70)
    print("SECTION 4.1: SAMPLE CHARACTERISTICS")
    print("=" * 70)

    n = len(df)
    mean = df['certainty'].mean()
    sd = df['certainty'].std()
    n_per_cond = n // df['influence_condition'].nunique()
    cells = df.groupby(['model_clean', 'product_id', 'influence_condition']).size()

    print(f"  N = {n:,}")
    print(f"  Design: {df['influence_condition'].nunique()}x{df['product_id'].nunique()}x{df['model_clean'].nunique()} factorial")
    print(f"  Trials per cell: {int(cells.mean())}")
    print(f"  Mean certainty: {mean:.2f}, SD: {sd:.2f}")
    print(f"  n per condition: {n_per_cond:,}")


def section_4_2(df):
    """Section 4.2: Primary Effects of Persuasive Content."""
    print("\n" + "=" * 70)
    print("SECTION 4.2: PRIMARY EFFECTS OF PERSUASIVE CONTENT")
    print("=" * 70)

    # Main ANOVA
    model = smf.ols('certainty ~ C(influence_condition)', data=df).fit()
    anova = anova_lm(model, typ=3)
    f_stat = anova.loc['C(influence_condition)', 'F']
    p_val = anova.loc['C(influence_condition)', 'PR(>F)']
    eta = compute_eta_squared(anova, 'C(influence_condition)')
    df_b = int(anova.loc['C(influence_condition)', 'df'])
    df_w = int(anova.loc['Residual', 'df'])

    print(f"\n  Main ANOVA: F({df_b}, {df_w}) = {f_stat:.3f}, p < .001, η² = {eta:.3f}")

    # Baseline differences
    baseline = df[df['influence_condition'] == 'control']
    h_base = baseline[baseline['category'] == 'hedonic']['certainty']
    u_base = baseline[baseline['category'] == 'utilitarian']['certainty']
    pooled = np.sqrt((h_base.std()**2 + u_base.std()**2) / 2)

    print(f"\n  Baseline (control only):")
    print(f"    Hedonic:     M = {h_base.mean():.2f}, SD = {h_base.std():.2f}")
    print(f"    Utilitarian: M = {u_base.mean():.2f}, SD = {u_base.std():.2f}")
    print(f"    Cohen's d = {(u_base.mean() - h_base.mean()) / pooled:.3f}")

    # Treatment effects
    ctrl = df[df['influence_condition'] == 'control']['certainty']
    print(f"\n  Control: M = {ctrl.mean():.2f}, SD = {ctrl.std():.2f}, n = {len(ctrl)}")
    print(f"\n  {'Treatment':<15} {'Effect':>8} {'95% CI':>22} {'Cohen d':>10} {'p':>12}")
    print(f"  {'-'*72}")

    for t in ['authority', 'social_proof', 'scarcity', 'reciprocity']:
        td = df[df['influence_condition'] == t]['certainty']
        eff = td.mean() - ctrl.mean()
        _, p = ttest_ind(td, ctrl)
        se = np.sqrt(ctrl.std()**2/len(ctrl) + td.std()**2/len(td))
        ci_lo, ci_hi = eff - 1.96*se, eff + 1.96*se
        d = eff / np.sqrt((ctrl.std()**2 + td.std()**2) / 2)
        sig = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"
        print(f"  {t:<15} {eff:>+8.3f} [{ci_lo:>7.3f}, {ci_hi:>7.3f}] {d:>10.3f} {p:>12.6f} {sig}")


def section_4_3(df):
    """Section 4.3: Product Category Interactions."""
    print("\n" + "=" * 70)
    print("SECTION 4.3: PRODUCT CATEGORY INTERACTIONS")
    print("=" * 70)

    formula = "certainty ~ C(influence_condition, Treatment('control')) * C(category, Treatment('utilitarian'))"
    model = smf.ols(formula, data=df).fit()
    anova = anova_lm(model, typ=3)
    key = "C(influence_condition, Treatment('control')):C(category, Treatment('utilitarian'))"
    f_int = anova.loc[key, 'F']
    p_int = anova.loc[key, 'PR(>F)']
    eta = compute_eta_squared(anova, key)
    df_n = int(anova.loc[key, 'df'])
    df_d = int(anova.loc['Residual', 'df'])

    print(f"\n  Interaction: F({df_n}, {df_d}) = {f_int:.3f}, p < .001, η² = {eta:.3f}")

    baseline = df[df['influence_condition'] == 'control']
    for cat in ['hedonic', 'utilitarian']:
        print(f"\n  --- {cat.upper()} ---")
        cb = baseline[baseline['category'] == cat]['certainty']
        print(f"  Baseline: M = {cb.mean():.3f}")
        print(f"  {'Treatment':<15} {'Effect':>8} {'95% CI':>24} {'p':>12}")
        print(f"  {'-'*65}")

        for t in ['authority', 'social_proof', 'scarcity', 'reciprocity']:
            td = df[(df['influence_condition'] == t) & (df['category'] == cat)]['certainty']
            eff = td.mean() - cb.mean()
            _, p = ttest_ind(td, cb)
            se = np.sqrt(cb.std()**2/len(cb) + td.std()**2/len(td))
            ci_lo, ci_hi = eff - 1.96*se, eff + 1.96*se
            sig = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"
            print(f"  {t:<15} {eff:>+8.3f} [{ci_lo:>8.3f}, {ci_hi:>7.3f}] {p:>12.6f} {sig}")


def section_4_4(df):
    """Section 4.4: Model-Specific Differences."""
    print("\n" + "=" * 70)
    print("SECTION 4.4: MODEL-SPECIFIC DIFFERENCES")
    print("=" * 70)

    model = smf.ols('certainty ~ C(model_clean)', data=df).fit()
    anova = anova_lm(model, typ=3)
    f_val = anova.loc['C(model_clean)', 'F']
    eta = compute_eta_squared(anova, 'C(model_clean)')
    df_n = int(anova.loc['C(model_clean)', 'df'])
    df_d = int(anova.loc['Residual', 'df'])

    print(f"\n  Between-model ANOVA: F({df_n}, {df_d}) = {f_val:.3f}, p < .001, η² = {eta:.3f}")
    print(f"\n  {'Model':<20} {'Mean':>8} {'SD':>8} {'n':>8}")
    print(f"  {'-'*48}")

    for m in ['gpt-4.1-mini', 'gpt-5-mini', 'kimi-k2-0905']:
        md = df[df['model_clean'] == m]['certainty']
        print(f"  {m:<20} {md.mean():>8.2f} {md.std():>8.2f} {len(md):>8}")

    g41 = df[df['model_clean'] == 'gpt-4.1-mini']['certainty']
    g5 = df[df['model_clean'] == 'gpt-5-mini']['certainty']
    n1, n2 = len(g41), len(g5)
    pooled = np.sqrt(((n1-1)*g41.var(ddof=1) + (n2-1)*g5.var(ddof=1)) / (n1+n2-2))
    print(f"\n  Cohen's d (GPT-4.1 Mini vs GPT-5 Mini): {(g41.mean()-g5.mean())/pooled:.3f}")


def section_4_5(df):
    """Section 4.5: Comprehensive Statistical Model (Table 2)."""
    print("\n" + "=" * 70)
    print("SECTION 4.5: COMPREHENSIVE STATISTICAL MODEL")
    print("=" * 70)

    # Focused model (without LLM)
    f_focused = "certainty ~ C(influence_condition, Treatment('control')) * C(category, Treatment('utilitarian'))"
    m_focused = smf.ols(f_focused, data=df).fit()
    print(f"\n  Focused model (no LLM): R² = {m_focused.rsquared:.3f}, F = {m_focused.fvalue:.3f}")

    # Full model
    f_full = f_focused + " + C(model_clean)"
    m_full = smf.ols(f_full, data=df).fit()
    print(f"  Full model:             R² = {m_full.rsquared:.3f}, F = {m_full.fvalue:.3f}")

    # Table 2: all effects from single full model
    anova = anova_lm(m_full, typ=3)
    df_r = int(anova.loc['Residual', 'df'])

    effects = {
        'Condition':        "C(influence_condition, Treatment('control'))",
        'Category':         "C(category, Treatment('utilitarian'))",
        'LLM Type':         "C(model_clean)",
        'Cond. x Category': "C(influence_condition, Treatment('control')):C(category, Treatment('utilitarian'))",
    }

    print(f"\n  TABLE 2: ANOVA Summary (Full Model)")
    print(f"  {'Source':<22} {'df':>12} {'F':>12} {'p':>12} {'η²':>8}")
    print(f"  {'-'*70}")

    for label, key in effects.items():
        row = anova.loc[key]
        f_val = row['F']
        p_val = row['PR(>F)']
        df_e = int(row['df'])
        eta = compute_eta_squared(anova, key)
        sig = "***" if p_val < .001 else "**" if p_val < .01 else "*" if p_val < .05 else "ns"
        p_str = "< 0.001" if p_val < .001 else f"  {p_val:.3f}"
        print(f"  {label:<22} {df_e:>3}, {df_r:>5} {f_val:>12.3f} {p_str:>12} {eta:>8.3f} {sig}")

    print(f"\n  Full model R² = {m_full.rsquared:.3f}, F = {m_full.fvalue:.3f}, p < .001")

    # Interaction coefficients (from focused model)
    print(f"\n  Interaction Coefficients (Condition x Hedonic):")
    print(f"  {'Treatment':<15} {'Coeff':>10} {'SE':>10} {'p':>12}")
    print(f"  {'-'*50}")

    for t in ['authority', 'social_proof', 'scarcity', 'reciprocity']:
        k = f"C(influence_condition, Treatment('control'))[T.{t}]:C(category, Treatment('utilitarian'))[T.hedonic]"
        if k in m_focused.params.index:
            print(f"  {t:<15} {m_focused.params[k]:>+10.3f} {m_focused.bse[k]:>10.3f} {m_focused.pvalues[k]:>12.6f}")


def section_4_6(df):
    """Section 4.6: Individual Product Analysis."""
    print("\n" + "=" * 70)
    print("SECTION 4.6: INDIVIDUAL PRODUCT ANALYSIS")
    print("=" * 70)

    baseline = df[df['influence_condition'] == 'control']
    products = ['concert_tickets', 'wine_tasting', 'spa_retreat',
                'laptop_computer', 'software_subscription', 'mobile_phone_plan']

    print(f"\n  {'Product':<25} {'Category':<12} {'High-cert rate':>15} {'n':>6}")
    print(f"  {'-'*62}")

    for p in products:
        pd_ = baseline[baseline['product_id'] == p]
        cat = pd_['category'].iloc[0]
        rate = (pd_['certainty'] >= 8.5).mean()
        print(f"  {p:<25} {cat:<12} {rate:>15.1%} {len(pd_):>6}")

    # Concert tickets + social proof
    cb = baseline[baseline['product_id'] == 'concert_tickets']
    cs = df[(df['influence_condition'] == 'social_proof') & (df['product_id'] == 'concert_tickets')]
    b_rate = (cb['certainty'] >= 8.5).mean()
    s_rate = (cs['certainty'] >= 8.5).mean()
    print(f"\n  Concert tickets + Social Proof:")
    print(f"    Baseline: {b_rate:.1%} -> Social Proof: {s_rate:.1%} ({(s_rate-b_rate)*100:+.1f} pp)")
    print(f"    Mean certainty: {cb['certainty'].mean():.2f} -> {cs['certainty'].mean():.2f}")


def section_4_7(df):
    """Section 4.7: Certainty Threshold Effects."""
    print("\n" + "=" * 70)
    print("SECTION 4.7: CERTAINTY THRESHOLD EFFECTS")
    print("=" * 70)

    baseline = df[df['influence_condition'] == 'control']
    h_rate = (baseline[baseline['category'] == 'hedonic']['certainty'] >= 8.5).mean()
    u_rate = (baseline[baseline['category'] == 'utilitarian']['certainty'] >= 8.5).mean()
    o_rate = (baseline['certainty'] >= 8.5).mean()

    print(f"\n  Baseline high-certainty rates (>= 8.5):")
    print(f"    Hedonic:     {h_rate:.1%}")
    print(f"    Utilitarian: {u_rate:.1%}")
    print(f"    Overall:     {o_rate:.1%}")

    hs = df[(df['influence_condition'] == 'social_proof') & (df['category'] == 'hedonic')]
    sr = (hs['certainty'] >= 8.5).mean()
    print(f"\n  Social proof on hedonic: {h_rate:.1%} -> {sr:.1%} ({(sr-h_rate)*100:+.1f} pp)")


def summary_table(df):
    """Summary statistics."""
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    for label, col in [("BY CONDITION", 'influence_condition'),
                       ("BY MODEL", 'model_clean'),
                       ("BY CATEGORY", 'category')]:
        print(f"\n  {label}:")
        print(f"  {'Group':<22} {'N':>6} {'Mean':>7} {'SD':>7} {'Rec%':>7}")
        print(f"  {'-'*52}")
        for g in df[col].unique():
            gd = df[df[col] == g]
            print(f"  {g:<22} {len(gd):>6} {gd['certainty'].mean():>7.2f} {gd['certainty'].std():>7.2f} {gd['recommendation'].mean():>7.1%}")


def main():
    print("=" * 70)
    print("Decision Gatekeepers: Statistical Analysis")
    print("=" * 70)

    try:
        df = load_data(FILE_PATH)
    except FileNotFoundError:
        print(f"\nERROR: {FILE_PATH} not found.")
        print("Place the dataset in the data/ directory.")
        return

    section_4_1(df)
    section_4_2(df)
    section_4_3(df)
    section_4_4(df)
    section_4_5(df)
    section_4_6(df)
    section_4_7(df)
    summary_table(df)

    print(f"\n{'=' * 70}")
    print(f"COMPLETE — {len(df):,} observations analysed.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
