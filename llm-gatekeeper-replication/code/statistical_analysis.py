"""
STATISTICAL ANALYSIS - LLMs as the Gatekeeper
==============================================
This script performs all statistical analyses reported in the paper
"LLMs as the Gatekeeper: Testing Persuasive Principles in AI Purchase Recommendations"

USAGE:
    python statistical_analysis.py

REQUIREMENTS:
    - pandas
    - numpy
    - scipy
    - statsmodels
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import f_oneway, ttest_ind
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
FILE_PATH = "data/llm_gatekeeper_dataset.csv"

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(file_path):
    """Load and prepare the experimental dataset"""
    
    print("=" * 80)
    print("DATA LOADING")
    print("=" * 80)
    
    df = pd.read_csv(file_path)
    print(f"\n Dataset loaded: {len(df):,} observations")
    
    # Show conditions
    print(f"\n   Conditions in dataset:")
    for cond in df['influence_condition'].unique():
        n = len(df[df['influence_condition'] == cond])
        print(f"      {cond}: {n:,}")
    
    # Clean model names for display
    df['model_clean'] = df['model_name'].str.replace('openai/', '').str.replace('moonshotai/', '')
    
    # Verify expected structure
    n_conditions = df['influence_condition'].nunique()
    n_products = df['product_id'].nunique()
    n_models = df['model_clean'].nunique()
    
    print(f"\n Experimental design: {n_conditions}×{n_products}×{n_models} factorial")
    print(f"   Expected observations: {n_conditions * n_products * n_models * 150:,}")
    print(f"   Actual observations: {len(df):,}")
    
    return df


# ============================================================================
# SECTION 4.1 - SAMPLE CHARACTERISTICS
# ============================================================================

def verify_section_4_1(df):
    """Verify basic descriptive statistics"""
    
    print("\n" + "=" * 80)
    print("SECTION 4.1: SAMPLE CHARACTERISTICS")
    print("=" * 80)
    
    # N total
    n_total = len(df)
    
    # Design
    n_conditions = df['influence_condition'].nunique()
    n_products = df['product_id'].nunique()
    n_models = df['model_clean'].nunique()
    
    # Trials per cell
    trials_per_cell = df.groupby(['model_clean', 'product_id', 'influence_condition']).size()
    
    # Overall statistics
    overall_mean = df['certainty'].mean()
    overall_sd = df['certainty'].std()
    
    # N per condition
    n_per_condition = len(df) // n_conditions
    
    print(f"\n EXPERIMENTAL DESIGN:")
    print(f"   N total: {n_total:,}")
    print(f"   Design: {n_conditions}×{n_products}×{n_models} factorial")
    print(f"   Trials per cell: {int(trials_per_cell.mean())} (min: {trials_per_cell.min()}, max: {trials_per_cell.max()})")
    
    print(f"\n OVERALL STATISTICS:")
    print(f"   Mean certainty: {overall_mean:.2f}")
    print(f"   SD certainty: {overall_sd:.2f}")
    print(f"   n per condition: {n_per_condition:,}")
    
    print(f"\n VERIFICATION:")
    print(f"   {'Statistic':<25} {'Calculated':>12} {'Expected':>12} {'Match':>8}")
    print(f"   {'-'*60}")
    print(f"   {'N total':<25} {n_total:>12,} {13500:>12,} {'✓' if n_total == 13500 else '✗':>8}")
    print(f"   {'Mean':<25} {overall_mean:>12.2f} {8.50:>12.2f} {'✓' if abs(overall_mean - 8.50) < 0.01 else '✗':>8}")
    print(f"   {'SD':<25} {overall_sd:>12.2f} {0.69:>12.2f} {'✓' if abs(overall_sd - 0.69) < 0.01 else '✗':>8}")
    
    return {
        'n_total': n_total,
        'overall_mean': overall_mean,
        'overall_sd': overall_sd
    }


# ============================================================================
# SECTION 4.2 - PRIMARY EFFECTS
# ============================================================================

def verify_section_4_2(df):
    """Verify primary treatment effects"""
    
    print("\n" + "=" * 80)
    print("SECTION 4.2: PRIMARY EFFECTS OF PERSUASIVE CONTENT")
    print("=" * 80)
    
    # --- MAIN ANOVA ---
    print("\n MAIN ANOVA (Effect of Condition):")
    
    groups = [df[df['influence_condition'] == cond]['certainty'].values 
              for cond in df['influence_condition'].unique()]
    
    f_stat, p_val = f_oneway(*groups)
    
    # Calculate η²
    grand_mean = df['certainty'].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
    ss_total = ((df['certainty'] - grand_mean)**2).sum()
    eta_squared = ss_between / ss_total
    
    df_between = len(groups) - 1
    df_within = len(df) - len(groups)
    
    print(f"   F({df_between}, {df_within}) = {f_stat:.3f}")
    print(f"   p < 0.001" if p_val < 0.001 else f"   p = {p_val:.6f}")
    print(f"   η² = {eta_squared:.3f}")
    
    # --- BASELINE DIFFERENCES (Hedonic vs Utilitarian) ---
    print("\n BASELINE DIFFERENCES (control only):")
    
    baseline_data = df[df['influence_condition'] == 'control']
    
    hedonic_baseline = baseline_data[baseline_data['category'] == 'hedonic']['certainty']
    util_baseline = baseline_data[baseline_data['category'] == 'utilitarian']['certainty']
    
    hedonic_mean = hedonic_baseline.mean()
    hedonic_sd = hedonic_baseline.std()
    util_mean = util_baseline.mean()
    util_sd = util_baseline.std()
    
    diff = util_mean - hedonic_mean
    t_stat_baseline, p_val_baseline = ttest_ind(util_baseline, hedonic_baseline)
    
    # Cohen's d
    pooled_sd = np.sqrt((hedonic_sd**2 + util_sd**2) / 2)
    cohens_d_baseline = diff / pooled_sd
    
    print(f"   Utilitarian: M = {util_mean:.2f}, SD = {util_sd:.2f}")
    print(f"   Hedonic: M = {hedonic_mean:.2f}, SD = {hedonic_sd:.2f}")
    print(f"   Difference: {diff:.2f} points")
    print(f"   t = {t_stat_baseline:.2f}, p < 0.001")
    print(f"   Cohen's d = {cohens_d_baseline:.3f}")
    
    # --- TREATMENT EFFECTS ---
    print("\n TREATMENT EFFECTS (vs control):")
    
    baseline_all = df[df['influence_condition'] == 'control']['certainty']
    baseline_mean = baseline_all.mean()
    baseline_sd = baseline_all.std()
    baseline_n = len(baseline_all)
    
    print(f"   Baseline (control): M = {baseline_mean:.2f}, SD = {baseline_sd:.2f}, n = {baseline_n}")
    
    treatments = ['authority', 'social_proof', 'scarcity', 'reciprocity']
    
    print(f"\n   {'Treatment':<15} {'Effect':>8} {'CI (95%)':>22} {'Cohen d':>10} {'p-value':>12}")
    print(f"   {'-'*75}")
    
    results = {}
    
    for treatment in treatments:
        treatment_data = df[df['influence_condition'] == treatment]['certainty']
        treatment_mean = treatment_data.mean()
        treatment_sd = treatment_data.std()
        treatment_n = len(treatment_data)
        
        # Effect
        effect = treatment_mean - baseline_mean
        
        # T-test
        t_stat, p_val = ttest_ind(treatment_data, baseline_all)
        
        # CI (using z = 1.96)
        se = np.sqrt((baseline_sd**2 / baseline_n) + (treatment_sd**2 / treatment_n))
        ci_lower = effect - 1.96 * se
        ci_upper = effect + 1.96 * se
        
        # Cohen's d
        pooled_sd_treat = np.sqrt((baseline_sd**2 + treatment_sd**2) / 2)
        cohens_d = effect / pooled_sd_treat
        
        # Significance
        if p_val < 0.001:
            sig = "***"
        elif p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"
        else:
            sig = "ns"
        
        results[treatment] = {
            'effect': effect,
            'ci': (ci_lower, ci_upper),
            'cohens_d': cohens_d,
            'p_value': p_val,
            'se': se
        }
        
        print(f"   {treatment:<15} {effect:>+8.3f} [{ci_lower:>7.3f}, {ci_upper:>7.3f}] {cohens_d:>10.3f} {p_val:>12.6f} {sig}")
    
    return results


# ============================================================================
# SECTION 4.3 - CATEGORY INTERACTIONS
# ============================================================================

def verify_section_4_3(df):
    """Verify category interactions"""
    
    print("\n" + "=" * 80)
    print("SECTION 4.3: CRITICAL PRODUCT CATEGORY INTERACTIONS")
    print("=" * 80)
    
    # --- INTERACTION ANOVA ---
    print("\n INTERACTION TEST (Condition × Category):")
    
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    
    formula = "certainty ~ C(influence_condition, Treatment('control')) * C(category, Treatment('utilitarian'))"
    model = smf.ols(formula, data=df).fit()
    anova_table = anova_lm(model, typ=3)
    
    # Extract interaction
    interaction_key = "C(influence_condition, Treatment('control')):C(category, Treatment('utilitarian'))"
    interaction_row = anova_table.loc[interaction_key]
    
    f_interaction = interaction_row['F']
    p_interaction = interaction_row['PR(>F)']
    df_num = int(interaction_row['df'])
    df_denom = int(anova_table.loc['Residual', 'df'])
    
    print(f"   F({df_num}, {df_denom}) = {f_interaction:.3f}")
    print(f"   p < 0.001" if p_interaction < 0.001 else f"   p = {p_interaction:.6f}")
    
    # --- EFFECTS BY CATEGORY ---
    print("\n EFFECTS BY PRODUCT CATEGORY:")
    
    treatments = ['authority', 'social_proof', 'scarcity', 'reciprocity']
    baseline_data = df[df['influence_condition'] == 'control']
    
    results = {'hedonic': {}, 'utilitarian': {}}
    
    for category in ['hedonic', 'utilitarian']:
        print(f"\n   --- {category.upper()} ---")
        
        cat_baseline = baseline_data[baseline_data['category'] == category]['certainty']
        baseline_mean = cat_baseline.mean()
        baseline_sd = cat_baseline.std()
        baseline_n = len(cat_baseline)
        
        print(f"   Baseline: M = {baseline_mean:.3f}")
        print(f"\n   {'Treatment':<15} {'Effect':>8} {'CI (95%)':>24} {'p-value':>12}")
        print(f"   {'-'*65}")
        
        for treatment in treatments:
            treatment_data = df[(df['influence_condition'] == treatment) & 
                               (df['category'] == category)]['certainty']
            
            treatment_mean = treatment_data.mean()
            treatment_sd = treatment_data.std()
            treatment_n = len(treatment_data)
            
            effect = treatment_mean - baseline_mean
            
            # T-test
            t_stat, p_val = ttest_ind(treatment_data, cat_baseline)
            
            # CI
            se = np.sqrt((baseline_sd**2 / baseline_n) + (treatment_sd**2 / treatment_n))
            ci_lower = effect - 1.96 * se
            ci_upper = effect + 1.96 * se
            
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            
            results[category][treatment] = {
                'effect': effect,
                'ci': (ci_lower, ci_upper),
                'p_value': p_val,
                'se': se
            }
            
            print(f"   {treatment:<15} {effect:>+8.3f} [{ci_lower:>8.3f}, {ci_upper:>7.3f}] {p_val:>12.6f} {sig}")
    
    return results


# ============================================================================
# SECTION 4.4 - MODEL-SPECIFIC DIFFERENCES
# ============================================================================

def verify_section_4_4(df):
    """Verify model differences"""
    
    print("\n" + "=" * 80)
    print("SECTION 4.4: MODEL-SPECIFIC DIFFERENCES")
    print("=" * 80)
    
    # --- BASELINE BY MODEL ---
    print("\n BASELINE CERTAINTY BY MODEL (control only):")
    
    baseline_data = df[df['influence_condition'] == 'control']
    
    models = ['gpt-4.1-mini', 'gpt-5-mini', 'kimi-k2-0905']
    
    print(f"\n   {'Model':<20} {'Mean':>8} {'SD':>8} {'n':>8}")
    print(f"   {'-'*50}")
    
    model_stats = {}
    
    for model in models:
        model_data = baseline_data[baseline_data['model_clean'] == model]['certainty']
        mean = model_data.mean()
        sd = model_data.std()
        n = len(model_data)
        
        model_stats[model] = {'mean': mean, 'sd': sd, 'n': n}
        
        print(f"   {model:<20} {mean:>8.2f} {sd:>8.2f} {n:>8}")
    
    # --- COHEN'S D BETWEEN MODELS ---
    print("\n📊 COHEN'S D BETWEEN MODELS:")
    
    gpt41 = baseline_data[baseline_data['model_clean'] == 'gpt-4.1-mini']['certainty']
    gpt5 = baseline_data[baseline_data['model_clean'] == 'gpt-5-mini']['certainty']
    
    pooled_sd = np.sqrt((gpt41.std()**2 + gpt5.std()**2) / 2)
    cohens_d = (gpt41.mean() - gpt5.mean()) / pooled_sd
    
    print(f"   GPT-4.1 Mini vs GPT-5 Mini: d = {cohens_d:.3f}")
    
    return model_stats


# ============================================================================
# SECTION 4.5 - COMPREHENSIVE MODEL
# ============================================================================

def verify_section_4_5(df):
    """Verify regression model"""
    
    print("\n" + "=" * 80)
    print("SECTION 4.5: COMPREHENSIVE STATISTICAL MODEL")
    print("=" * 80)
    
    import statsmodels.formula.api as smf
    
    # --- FOCUSED MODEL (Condition × Category only) ---
    print("\n FOCUSED MODEL (Condition × Category):")
    
    formula_focused = """certainty ~ C(influence_condition, Treatment('control')) * 
                                      C(category, Treatment('utilitarian'))"""
    
    model_focused = smf.ols(formula_focused, data=df).fit()
    
    r2_focused = model_focused.rsquared
    f_focused = model_focused.fvalue
    
    print(f"   R² = {r2_focused:.3f}")
    print(f"   F = {f_focused:.3f}")
    
    # --- FULL MODEL (including LLM) ---
    print("\n FULL MODEL (including LLM):")
    
    formula_full = """certainty ~ C(influence_condition, Treatment('control')) * 
                                   C(category, Treatment('utilitarian')) + 
                                   C(model_clean)"""
    
    model_full = smf.ols(formula_full, data=df).fit()
    
    r2_full = model_full.rsquared
    f_full = model_full.fvalue
    
    print(f"   R² = {r2_full:.3f}")
    print(f"   F = {f_full:.3f}")
    
    # --- INTERACTION COEFFICIENTS ---
    print("\n INTERACTION COEFFICIENTS (Condition × Hedonic):")
    
    treatments = ['authority', 'social_proof', 'scarcity', 'reciprocity']
    
    print(f"\n   {'Treatment':<15} {'Coefficient':>12} {'Std Err':>12} {'p-value':>12}")
    print(f"   {'-'*55}")
    
    for treatment in treatments:
        coef_name = f"C(influence_condition, Treatment('control'))[T.{treatment}]:C(category, Treatment('utilitarian'))[T.hedonic]"
        
        if coef_name in model_focused.params.index:
            coef = model_focused.params[coef_name]
            se = model_focused.bse[coef_name]
            pval = model_focused.pvalues[coef_name]
            print(f"   {treatment:<15} {coef:>+12.3f} {se:>12.3f} {pval:>12.6f}")
    
    return {
        'r2_focused': r2_focused,
        'r2_full': r2_full,
        'f_full': f_full
    }


# ============================================================================
# SECTION 4.6 - INDIVIDUAL PRODUCTS
# ============================================================================

def verify_section_4_6(df):
    """Verify individual product analysis"""
    
    print("\n" + "=" * 80)
    print("SECTION 4.6: INDIVIDUAL PRODUCT ANALYSIS")
    print("=" * 80)
    
    # --- BASELINE RATES BY PRODUCT ---
    print("\n BASELINE HIGH-CERTAINTY RATES (≥8.5):")
    
    baseline_data = df[df['influence_condition'] == 'control']
    
    products = {
        'concert_tickets': 'hedonic',
        'wine_tasting': 'hedonic', 
        'spa_retreat': 'hedonic',
        'laptop_computer': 'utilitarian',
        'software_subscription': 'utilitarian',
        'mobile_phone_plan': 'utilitarian'
    }
    
    print(f"\n   {'Product':<25} {'Category':<12} {'Rate':>10} {'n':>8}")
    print(f"   {'-'*60}")
    
    for product, category in products.items():
        product_data = baseline_data[baseline_data['product_id'] == product]
        rate = (product_data['certainty'] >= 8.5).mean()
        n = len(product_data)
        
        print(f"   {product:<25} {category:<12} {rate:>10.1%} {n:>8}")
    
    # --- CONCERT TICKETS + SOCIAL PROOF ---
    print("\n CONCERT TICKETS + SOCIAL PROOF (dramatic effect):")
    
    concert_baseline = baseline_data[baseline_data['product_id'] == 'concert_tickets']
    concert_social = df[(df['influence_condition'] == 'social_proof') & 
                        (df['product_id'] == 'concert_tickets')]
    
    baseline_rate = (concert_baseline['certainty'] >= 8.5).mean()
    social_rate = (concert_social['certainty'] >= 8.5).mean()
    change = social_rate - baseline_rate
    
    print(f"   Baseline: {baseline_rate:.1%}")
    print(f"   After Social Proof: {social_rate:.1%}")
    print(f"   Change: {change:+.1%} ({change*100:+.1f} percentage points)")
    
    return {
        'concert_baseline': baseline_rate,
        'concert_social': social_rate
    }


# ============================================================================
# SECTION 4.7 - THRESHOLD EFFECTS
# ============================================================================

def verify_section_4_7(df):
    """Verify threshold effects"""
    
    print("\n" + "=" * 80)
    print("SECTION 4.7: CERTAINTY THRESHOLD EFFECTS")
    print("=" * 80)
    
    baseline_data = df[df['influence_condition'] == 'control']
    
    # --- BASELINE BY CATEGORY ---
    print("\n BASELINE HIGH-CERTAINTY RATES (≥8.5) BY CATEGORY:")
    
    hedonic_baseline = baseline_data[baseline_data['category'] == 'hedonic']
    util_baseline = baseline_data[baseline_data['category'] == 'utilitarian']
    overall_baseline = baseline_data
    
    hedonic_rate = (hedonic_baseline['certainty'] >= 8.5).mean()
    util_rate = (util_baseline['certainty'] >= 8.5).mean()
    overall_rate = (overall_baseline['certainty'] >= 8.5).mean()
    
    print(f"   Hedonic: {hedonic_rate:.1%}")
    print(f"   Utilitarian: {util_rate:.1%}")
    print(f"   Overall: {overall_rate:.1%}")
    
    # --- SOCIAL PROOF EFFECT ON HEDONIC ---
    print("\n SOCIAL PROOF EFFECT ON HEDONIC (threshold):")
    
    hedonic_social = df[(df['influence_condition'] == 'social_proof') & 
                        (df['category'] == 'hedonic')]
    
    social_rate = (hedonic_social['certainty'] >= 8.5).mean()
    
    print(f"   Baseline: {hedonic_rate:.1%}")
    print(f"   After Social Proof: {social_rate:.1%}")
    print(f"   Change: {social_rate - hedonic_rate:+.1%}")
    
    return {
        'hedonic_rate': hedonic_rate,
        'util_rate': util_rate,
        'overall_rate': overall_rate
    }


# ============================================================================
# SUMMARY STATISTICS TABLE
# ============================================================================

def generate_summary_table(df):
    """Generate summary statistics table for paper"""
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS TABLE")
    print("=" * 80)
    
    print("\n BY CONDITION:")
    print(f"\n   {'Condition':<15} {'N':>8} {'Mean':>8} {'SD':>8} {'Rec Rate':>10}")
    print(f"   {'-'*55}")
    
    for condition in ['control', 'authority', 'social_proof', 'scarcity', 'reciprocity']:
        cond_data = df[df['influence_condition'] == condition]
        n = len(cond_data)
        mean = cond_data['certainty'].mean()
        sd = cond_data['certainty'].std()
        rec_rate = cond_data['recommendation'].mean()
        print(f"   {condition:<15} {n:>8} {mean:>8.2f} {sd:>8.2f} {rec_rate:>10.1%}")
    
    print("\n BY MODEL:")
    print(f"\n   {'Model':<20} {'N':>8} {'Mean':>8} {'SD':>8} {'Rec Rate':>10}")
    print(f"   {'-'*60}")
    
    for model in df['model_clean'].unique():
        model_data = df[df['model_clean'] == model]
        n = len(model_data)
        mean = model_data['certainty'].mean()
        sd = model_data['certainty'].std()
        rec_rate = model_data['recommendation'].mean()
        print(f"   {model:<20} {n:>8} {mean:>8.2f} {sd:>8.2f} {rec_rate:>10.1%}")
    
    print("\n BY CATEGORY:")
    print(f"\n   {'Category':<15} {'N':>8} {'Mean':>8} {'SD':>8} {'Rec Rate':>10}")
    print(f"   {'-'*55}")
    
    for category in ['hedonic', 'utilitarian']:
        cat_data = df[df['category'] == category]
        n = len(cat_data)
        mean = cat_data['certainty'].mean()
        sd = cat_data['certainty'].std()
        rec_rate = cat_data['recommendation'].mean()
        print(f"   {category:<15} {n:>8} {mean:>8.2f} {sd:>8.2f} {rec_rate:>10.1%}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run complete statistical analysis"""
    
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS")
    print("LLMs as the Gatekeeper: Testing Persuasive Principles")
    print("in AI Purchase Recommendations")
    print("=" * 80)
    
    # Load data
    try:
        df = load_data(FILE_PATH)
    except FileNotFoundError:
        print(f"\n❌ ERROR: File not found:")
        print(f"   {FILE_PATH}")
        print(f"\nPlease ensure the dataset is in the correct location.")
        return
    
    # Run all analyses
    verify_section_4_1(df)
    verify_section_4_2(df)
    verify_section_4_3(df)
    verify_section_4_4(df)
    verify_section_4_5(df)
    verify_section_4_6(df)
    verify_section_4_7(df)
    
    # Generate summary table
    generate_summary_table(df)
    
    # Final summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nTotal observations analyzed: {len(df):,}")
    print(f"All statistics calculated from raw data.")


if __name__ == "__main__":
    main()
