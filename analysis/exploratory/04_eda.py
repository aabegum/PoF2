"""
EXPLORATORY DATA ANALYSIS (EDA) - POF PREDICTION
Turkish EDAŞ PoF Prediction Project

Purpose:
- Analyze engineered features and patterns
- Visualize failure behaviors
- Generate insights for model interpretation

Input:  data/features_engineered.csv (from 03_feature_engineering.py)
Output: outputs/eda/*.png, reports/eda_summary.txt

Author: Data Analytics Team
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import sys
# Fix Unicode encoding for Windows console (Turkish cp1254 issue)
if sys.platform == 'win32':
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleCP(65001)
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
warnings.filterwarnings('ignore')

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("="*100)
print(" "*30 + "EXPLORATORY DATA ANALYSIS")
print(" "*25 + "Turkish EDAŞ PoF Prediction Project")
print("="*100)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Create output directories
Path('outputs/eda').mkdir(parents=True, exist_ok=True)
Path('reports').mkdir(exist_ok=True)

# Output file for text summary
report_file = open('reports/eda_summary.txt', 'w', encoding='utf-8')

def log_print(text):
    """Print to console and write to report file"""
    print(text)
    report_file.write(text + '\n')

log_print("\n📋 EDA Configuration:")
log_print(f"   Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_print(f"   Output Directory: outputs/eda/")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
log_print("\n" + "="*100)
log_print("STEP 1: LOADING DATA")
log_print("="*100)

data_path = Path('data/features_engineered.csv')

if not data_path.exists():
    log_print(f"\n❌ ERROR: File not found at {data_path}")
    log_print("Please run 03_feature_engineering.py first!")
    report_file.close()
    exit(1)

log_print(f"\n✓ Loading from: {data_path}")
df = pd.read_csv(data_path)
log_print(f"✓ Loaded: {df.shape[0]:,} equipment × {df.shape[1]} features")

# Verify Equipment_Class_Primary exists (created by 02_data_transformation.py v2.0+)
if 'Equipment_Class_Primary' not in df.columns:
    log_print("\n⚠ WARNING: Equipment_Class_Primary column not found!")
    log_print("This column should be created by 02_data_transformation.py and passed through 03_feature_engineering.py")
    log_print("Some visualizations may be limited without this column.")
else:
    log_print("✓ Equipment_Class_Primary column verified (from transformation script)")

# Display column names
log_print(f"\n--- Available Features ({df.shape[1]}) ---")
for i, col in enumerate(df.columns, 1):
    log_print(f"  {i:2d}. {col}")

# ============================================================================
# STEP 2: DATA OVERVIEW
# ============================================================================
log_print("\n" + "="*100)
log_print("STEP 2: DATA OVERVIEW & QUALITY")
log_print("="*100)

# Basic info
log_print(f"\n--- Dataset Dimensions ---")
log_print(f"Total Equipment: {df.shape[0]:,}")
log_print(f"Total Features: {df.shape[1]:,}")
log_print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Column types
log_print(f"\n--- Data Types ---")
dtype_counts = df.dtypes.value_counts()
for dtype, count in dtype_counts.items():
    log_print(f"  {dtype}: {count} features")

# Identify column categories
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
date_cols = [col for col in df.columns if 'tarih' in col.lower() or 'date' in col.lower()]

log_print(f"\n--- Feature Categories ---")
log_print(f"  Numeric features: {len(numeric_cols)}")
log_print(f"  Categorical features: {len(categorical_cols)}")
log_print(f"  Date features: {len(date_cols)}")

# Missing values analysis
log_print(f"\n--- Missing Values Analysis ---")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Feature': missing.index,
    'Missing_Count': missing.values,
    'Missing_Percent': missing_pct.values
})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

if len(missing_df) > 0:
    log_print(f"\nFeatures with missing values: {len(missing_df)}")
    log_print(f"\nTop 20 features by missing values:")
    log_print(missing_df.head(20).to_string(index=False))
    
    # Visualize missing values
    if len(missing_df) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        top_missing = missing_df.head(20)
        ax.barh(range(len(top_missing)), top_missing['Missing_Percent'].values, color='coral')
        ax.set_yticks(range(len(top_missing)))
        ax.set_yticklabels(top_missing['Feature'].values, fontsize=9)
        ax.set_xlabel('Missing Percentage (%)', fontsize=11)
        ax.set_title('Top 20 Features with Missing Values', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/eda/01_missing_values.png', dpi=300, bbox_inches='tight')
        plt.close()
        log_print("\n✓ Saved: outputs/eda/01_missing_values.png")
else:
    log_print("\n✓ No missing values found!")

# ============================================================================
# STEP 3: EQUIPMENT CLASS ANALYSIS
# ============================================================================
log_print("\n" + "="*100)
log_print("STEP 3: EQUIPMENT CLASS ANALYSIS")
log_print("="*100)

if 'Equipment_Class_Primary' in df.columns:
    log_print(f"\n--- Equipment Class Distribution ---")
    class_dist = df['Equipment_Class_Primary'].value_counts()
    
    log_print(f"\nTotal Equipment Classes: {len(class_dist)}")
    log_print(f"\nAll Equipment Classes:")
    for i, (cls, count) in enumerate(class_dist.items(), 1):
        pct = count / len(df) * 100
        log_print(f"  {i:2d}. {cls:<30} {count:>6,} ({pct:>5.1f}%)")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart
    ax = axes[0]
    class_dist.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('Number of Equipment', fontsize=11)
    ax.set_ylabel('Equipment Class', fontsize=11)
    ax.set_title('Equipment Class Distribution', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # Pie chart (top 10 + others)
    ax = axes[1]
    if len(class_dist) > 10:
        top_10 = class_dist.head(10)
        others = class_dist[10:].sum()
        pie_data = list(top_10.values) + [others]
        pie_labels = list(top_10.index) + ['Others']
    else:
        pie_data = class_dist.values
        pie_labels = class_dist.index
    
    ax.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', startangle=90)
    ax.set_title('Equipment Class Distribution', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/eda/02_equipment_classes.png', dpi=300, bbox_inches='tight')
    plt.close()
    log_print("\n✓ Saved: outputs/eda/02_equipment_classes.png")

# ============================================================================
# STEP 4: FAILURE RATE ANALYSIS
# ============================================================================
log_print("\n" + "="*100)
log_print("STEP 4: FAILURE RATE ANALYSIS")
log_print("="*100)

# Analyze failure counts for different horizons
failure_horizons = {
    '3M': 'Arıza_Sayısı_3ay',
    '6M': 'Arıza_Sayısı_6ay',
    '12M': 'Arıza_Sayısı_12ay',
    'Lifetime': 'Toplam_Arıza_Sayisi_Lifetime'
}

available_horizons = {k: v for k, v in failure_horizons.items() if v in df.columns}

if available_horizons:
    log_print(f"\n--- Failure Statistics by Horizon ---")
    
    failure_stats = []
    
    for horizon, col in available_horizons.items():
        # Binary: equipment with at least 1 failure
        has_failure = (df[col] > 0).sum()
        no_failure = (df[col] == 0).sum()
        failure_rate = has_failure / len(df) * 100
        
        # Statistics
        stats = df[col].describe()
        
        log_print(f"\n{horizon} ({col}):")
        log_print(f"  Equipment with failures: {has_failure:,} ({failure_rate:.1f}%)")
        log_print(f"  Equipment without failures: {no_failure:,} ({100-failure_rate:.1f}%)")
        log_print(f"  Mean failures per equipment: {stats['mean']:.2f}")
        log_print(f"  Median failures: {stats['50%']:.0f}")
        log_print(f"  Max failures: {stats['max']:.0f}")
        
        failure_stats.append({
            'Horizon': horizon,
            'With_Failures': has_failure,
            'Without_Failures': no_failure,
            'Failure_Rate': failure_rate
        })
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Failure rates comparison
    ax = axes[0, 0]
    horizons_list = [s['Horizon'] for s in failure_stats]
    failure_rates = [s['Failure_Rate'] for s in failure_stats]
    ax.bar(horizons_list, failure_rates, color='coral', alpha=0.7)
    ax.set_ylabel('Failure Rate (%)', fontsize=11)
    ax.set_title('Equipment Failure Rate by Horizon', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, (h, rate) in enumerate(zip(horizons_list, failure_rates)):
        ax.text(i, rate, f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Distribution of failure counts (12M)
    if '12M' in available_horizons:
        ax = axes[0, 1]
        col = available_horizons['12M']
        data = df[col][df[col] > 0]  # Only equipment with failures
        ax.hist(data, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel('Number of Failures (12M)', fontsize=11)
        ax.set_ylabel('Number of Equipment', fontsize=11)
        ax.set_title('Distribution of Failure Counts (12M)', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)
    
    # Failure by equipment class (12M)
    if '12M' in available_horizons and 'Equipment_Class_Primary' in df.columns:
        ax = axes[1, 0]
        col = available_horizons['12M']
        class_failure = df.groupby('Equipment_Class_Primary')[col].agg(['mean', 'count'])
        class_failure = class_failure.sort_values('mean', ascending=False).head(15)
        
        ax.barh(range(len(class_failure)), class_failure['mean'].values, color='lightcoral')
        ax.set_yticks(range(len(class_failure)))
        ax.set_yticklabels(class_failure.index, fontsize=9)
        ax.set_xlabel('Average Failures (12M)', fontsize=11)
        ax.set_title('Average 12M Failures by Equipment Class (Top 15)', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
    
    # Cumulative failure distribution
    if 'Lifetime' in available_horizons:
        ax = axes[1, 1]
        col = available_horizons['Lifetime']
        data = df[col].sort_values()
        cumulative = np.arange(1, len(data) + 1) / len(data) * 100
        ax.plot(data, cumulative, linewidth=2, color='purple')
        ax.set_xlabel('Number of Lifetime Failures', fontsize=11)
        ax.set_ylabel('Cumulative % of Equipment', fontsize=11)
        ax.set_title('Cumulative Failure Distribution (Lifetime)', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Add reference lines
        ax.axhline(50, color='red', linestyle='--', alpha=0.5, label='Median')
        ax.axhline(80, color='orange', linestyle='--', alpha=0.5, label='80th percentile')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('outputs/eda/03_failure_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    log_print("\n✓ Saved: outputs/eda/03_failure_analysis.png")

# ============================================================================
# STEP 5: EQUIPMENT AGE ANALYSIS
# ============================================================================
log_print("\n" + "="*100)
log_print("STEP 5: EQUIPMENT AGE ANALYSIS")
log_print("="*100)

age_cols = ['Ekipman_Yaşı_Yıl', 'Beklenen_Ömür_Yıl', 'Yas_Beklenen_Omur_Orani']
available_age_cols = [col for col in age_cols if col in df.columns]

if available_age_cols:
    log_print(f"\n--- Age Statistics ---")
    
    for col in available_age_cols:
        stats = df[col].describe()
        log_print(f"\n{col}:")
        log_print(f"  Mean: {stats['mean']:.2f}")
        log_print(f"  Median: {stats['50%']:.2f}")
        log_print(f"  Min: {stats['min']:.2f}")
        log_print(f"  Max: {stats['max']:.2f}")
    
    # Visualize
    n_plots = len(available_age_cols)
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    
    if n_plots == 1:
        axes = [axes]
    
    for idx, col in enumerate(available_age_cols):
        ax = axes[idx]
        data = df[col].dropna()
        ax.hist(data, bins=40, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel('Number of Equipment', fontsize=11)
        ax.set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
        ax.axvline(data.median(), color='red', linestyle='--', 
                   label=f'Median: {data.median():.1f}')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/eda/04_age_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    log_print("\n✓ Saved: outputs/eda/04_age_analysis.png")
    
    # Age vs Failure relationship
    if 'Ekipman_Yaşı_Yıl' in df.columns and 'Arıza_Sayısı_12ay' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create age bins
        df['Age_Bin'] = pd.cut(df['Ekipman_Yaşı_Yıl'], bins=10)
        age_failure = df.groupby('Age_Bin')['Arıza_Sayısı_12ay'].mean()
        
        x_labels = [f'{int(interval.left)}-{int(interval.right)}' for interval in age_failure.index]
        ax.bar(range(len(age_failure)), age_failure.values, color='coral', alpha=0.7)
        ax.set_xticks(range(len(age_failure)))
        ax.set_xticklabels(x_labels, rotation=45)
        ax.set_xlabel('Equipment Age (Years)', fontsize=11)
        ax.set_ylabel('Average Failures (12M)', fontsize=11)
        ax.set_title('Relationship: Equipment Age vs Failure Rate', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/eda/05_age_vs_failure.png', dpi=300, bbox_inches='tight')
        plt.close()
        log_print("\n✓ Saved: outputs/eda/05_age_vs_failure.png")

# ============================================================================
# STEP 6: RELIABILITY METRICS
# ============================================================================
log_print("\n" + "="*100)
log_print("STEP 6: RELIABILITY METRICS ANALYSIS")
log_print("="*100)

reliability_cols = ['MTBF_Gün', 'Reliability_Score', 'Son_Arıza_Gun_Sayisi']
available_reliability = [col for col in reliability_cols if col in df.columns]

if available_reliability:
    log_print(f"\n--- Reliability Metrics Statistics ---")
    
    for col in available_reliability:
        stats = df[col].describe()
        log_print(f"\n{col}:")
        log_print(f"  Count: {stats['count']:.0f}")
        log_print(f"  Mean: {stats['mean']:.2f}")
        log_print(f"  Median: {stats['50%']:.2f}")
        log_print(f"  Std Dev: {stats['std']:.2f}")
        log_print(f"  Min: {stats['min']:.2f}")
        log_print(f"  Max: {stats['max']:.2f}")
    
    # Visualize
    n_plots = len(available_reliability)
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    
    if n_plots == 1:
        axes = [axes]
    
    for idx, col in enumerate(available_reliability):
        ax = axes[idx]
        data = df[col].dropna()
        ax.hist(data, bins=50, edgecolor='black', alpha=0.7, color='lightgreen')
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'Distribution of {col}', fontsize=11, fontweight='bold')
        ax.axvline(data.median(), color='red', linestyle='--', 
                   label=f'Median: {data.median():.1f}')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/eda/06_reliability_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    log_print("\n✓ Saved: outputs/eda/06_reliability_metrics.png")

# ============================================================================
# STEP 7: GEOGRAPHIC ANALYSIS
# ============================================================================
log_print("\n" + "="*100)
log_print("STEP 7: GEOGRAPHIC ANALYSIS")
log_print("="*100)

if 'Geographic_Cluster' in df.columns:
    log_print(f"\n--- Geographic Cluster Distribution ---")
    cluster_dist = df['Geographic_Cluster'].value_counts().sort_index()
    
    log_print(f"\nTotal Clusters: {len(cluster_dist)}")
    log_print(f"\nCluster Sizes:")
    for cluster, count in cluster_dist.items():
        pct = count / len(df) * 100
        log_print(f"  Cluster {cluster}: {count:,} equipment ({pct:.1f}%)")
    
    # Cluster vs Failure analysis
    if 'Arıza_Sayısı_12ay' in df.columns:
        cluster_failure = df.groupby('Geographic_Cluster')['Arıza_Sayısı_12ay'].agg(['mean', 'count'])
        
        log_print(f"\n--- Average 12M Failures by Cluster ---")
        for cluster in cluster_failure.index:
            log_print(f"  Cluster {cluster}: {cluster_failure.loc[cluster, 'mean']:.2f} failures/equipment")
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Cluster sizes
        ax = axes[0]
        ax.bar(cluster_dist.index, cluster_dist.values, color='steelblue', alpha=0.7)
        ax.set_xlabel('Geographic Cluster', fontsize=11)
        ax.set_ylabel('Number of Equipment', fontsize=11)
        ax.set_title('Equipment Distribution by Geographic Cluster', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Cluster failure rates
        ax = axes[1]
        ax.bar(cluster_failure.index, cluster_failure['mean'].values, color='coral', alpha=0.7)
        ax.set_xlabel('Geographic Cluster', fontsize=11)
        ax.set_ylabel('Average Failures (12M)', fontsize=11)
        ax.set_title('Average Failure Rate by Geographic Cluster', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/eda/07_geographic_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        log_print("\n✓ Saved: outputs/eda/07_geographic_analysis.png")

# ============================================================================
# STEP 8: CORRELATION ANALYSIS
# ============================================================================
log_print("\n" + "="*100)
log_print("STEP 8: CORRELATION ANALYSIS (KEY FEATURES)")
log_print("="*100)

# Select key features for correlation
key_features = [
    'Ekipman_Yaşı_Yıl', 'Yas_Beklenen_Omur_Orani', 
    'MTBF_Gün', 'Reliability_Score',
    'Arıza_Sayısı_3ay', 'Arıza_Sayısı_6ay', 'Arıza_Sayısı_12ay',
    'Toplam_Arıza_Sayisi_Lifetime', 'Son_Arıza_Gun_Sayisi'
]

available_key_features = [f for f in key_features if f in df.columns]

if len(available_key_features) > 1:
    log_print(f"\nAnalyzing correlations between {len(available_key_features)} key features")
    
    # Calculate correlation matrix
    corr_matrix = df[available_key_features].corr()
    
    # Find high correlations
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                high_corr_pairs.append({
                    'Feature_1': corr_matrix.columns[i],
                    'Feature_2': corr_matrix.columns[j],
                    'Correlation': corr_val
                })
    
    high_corr_pairs = sorted(high_corr_pairs, key=lambda x: abs(x['Correlation']), reverse=True)
    
    if high_corr_pairs:
        log_print(f"\n--- High Correlations (|r| > 0.5) ---")
        log_print(f"Found {len(high_corr_pairs)} pairs:")
        for pair in high_corr_pairs[:15]:
            log_print(f"  {pair['Feature_1']} <-> {pair['Feature_2']}: {pair['Correlation']:.3f}")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                ax=ax, vmin=-1, vmax=1)
    ax.set_title('Correlation Matrix - Key Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/eda/08_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    log_print("\n✓ Saved: outputs/eda/08_correlation_matrix.png")

# ============================================================================
# STEP 9: RISK CATEGORY ANALYSIS
# ============================================================================
log_print("\n" + "="*100)
log_print("STEP 9: RISK CATEGORY ANALYSIS")
log_print("="*100)

if 'Risk_Category' in df.columns:
    log_print(f"\n--- Risk Category Distribution ---")
    risk_dist = df['Risk_Category'].value_counts()
    
    for risk, count in risk_dist.items():
        pct = count / len(df) * 100
        log_print(f"  {risk}: {count:,} ({pct:.1f}%)")
    
    # Risk vs Failure
    if 'Arıza_Sayısı_12ay' in df.columns:
        risk_failure = df.groupby('Risk_Category')['Arıza_Sayısı_12ay'].agg(['mean', 'median', 'count'])
        
        log_print(f"\n--- 12M Failures by Risk Category ---")
        for risk in risk_failure.index:
            log_print(f"  {risk}:")
            log_print(f"    Mean: {risk_failure.loc[risk, 'mean']:.2f}")
            log_print(f"    Median: {risk_failure.loc[risk, 'median']:.0f}")
            log_print(f"    Equipment count: {risk_failure.loc[risk, 'count']:.0f}")
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Distribution
        ax = axes[0]
        risk_dist.plot(kind='bar', ax=ax, color='steelblue', alpha=0.7)
        ax.set_xlabel('Risk Category', fontsize=11)
        ax.set_ylabel('Number of Equipment', fontsize=11)
        ax.set_title('Equipment Distribution by Risk Category', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Risk vs Failure
        ax = axes[1]
        risk_failure['mean'].plot(kind='bar', ax=ax, color='coral', alpha=0.7)
        ax.set_xlabel('Risk Category', fontsize=11)
        ax.set_ylabel('Average Failures (12M)', fontsize=11)
        ax.set_title('Average 12M Failures by Risk Category', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig('outputs/eda/09_risk_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        log_print("\n✓ Saved: outputs/eda/09_risk_analysis.png")

# ============================================================================
# STEP 10: MODEL RESULTS ANALYSIS (IF AVAILABLE)
# ============================================================================
log_print("\n" + "="*100)
log_print("STEP 10: MODEL PREDICTIONS ANALYSIS (IF AVAILABLE)")
log_print("="*100)

# Check for different prediction sources
predictions_dir = Path('predictions')
results_dir = Path('results')

found_any_predictions = False

# 1. Check for survival analysis predictions (Model 1 - Temporal PoF)
pof_multi_horizon_path = predictions_dir / 'pof_multi_horizon_predictions.csv'
if pof_multi_horizon_path.exists():
    log_print(f"\n--- MODEL 1: TEMPORAL POF PREDICTIONS (Survival Analysis) ---")
    log_print(f"Found: {pof_multi_horizon_path}")

    pred_df = pd.read_csv(pof_multi_horizon_path)
    log_print(f"Equipment: {len(pred_df):,}")

    # Analyze multi-horizon predictions
    for horizon in ['3M', '12M', '24M']:
        pof_col = f'PoF_Probability_{horizon}'
        if pof_col in pred_df.columns:
            mean_pof = pred_df[pof_col].mean()
            median_pof = pred_df[pof_col].median()
            log_print(f"\n{horizon} Predictions:")
            log_print(f"  Mean PoF: {mean_pof:.2%}")
            log_print(f"  Median PoF: {median_pof:.2%}")
            log_print(f"  High risk (>40%): {(pred_df[pof_col] > 0.4).sum():,}")

    # Check Risk_Category distribution
    if 'Risk_Category' in pred_df.columns:
        log_print(f"\nRisk Category Distribution:")
        risk_dist = pred_df['Risk_Category'].value_counts()
        for cat, count in risk_dist.items():
            pct = count / len(pred_df) * 100
            log_print(f"  {cat}: {count:,} ({pct:.1f}%)")

    found_any_predictions = True

# 2. Check for risk assessment files (Model 1 + CoF)
risk_files = {
    '3M': results_dir / 'risk_assessment_3M.csv',
    '12M': results_dir / 'risk_assessment_12M.csv',
    '24M': results_dir / 'risk_assessment_24M.csv'
}

for horizon, risk_path in risk_files.items():
    if risk_path.exists():
        if not found_any_predictions:
            log_print(f"\n--- RISK ASSESSMENT (PoF × CoF) ---")

        log_print(f"\nFound: {risk_path}")
        risk_df = pd.read_csv(risk_path)

        log_print(f"{horizon} Risk Assessment:")
        log_print(f"  Equipment: {len(risk_df):,}")

        if 'Risk_Score' in risk_df.columns:
            mean_risk = risk_df['Risk_Score'].mean()
            log_print(f"  Mean Risk Score: {mean_risk:.1f}/100")

        if 'Risk_Category' in risk_df.columns:
            risk_cat_dist = risk_df['Risk_Category'].value_counts()
            for cat, count in risk_cat_dist.items():
                pct = count / len(risk_df) * 100
                log_print(f"    {cat}: {count:,} ({pct:.1f}%)")

        found_any_predictions = True

# 3. Check for CAPEX priority list
capex_path = results_dir / 'capex_priority_list.csv'
if capex_path.exists():
    log_print(f"\n--- CAPEX PRIORITY LIST ---")
    log_print(f"Found: {capex_path}")

    capex_df = pd.read_csv(capex_path)
    log_print(f"Top priority equipment: {len(capex_df):,}")

    if 'Recommended_Action' in capex_df.columns:
        action_dist = capex_df['Recommended_Action'].value_counts()
        log_print(f"\nRecommended Actions:")
        for action, count in action_dist.items():
            log_print(f"  {action}: {count:,}")

    found_any_predictions = True

# 4. Check for legacy Model 2 predictions (chronic repeater)
if predictions_dir.exists():
    pred_files = list(predictions_dir.glob('failure_predictions_*.csv'))

    if pred_files:
        log_print(f"\n--- MODEL 2: CHRONIC REPEATER PREDICTIONS ---")
        log_print(f"Found {len(pred_files)} prediction files")

        for pred_file in pred_files:
            pred_df = pd.read_csv(pred_file)
            horizon = pred_file.stem.split('_')[-1]

            log_print(f"\n{horizon.upper()} Predictions ({pred_file.name}):")
            log_print(f"  Equipment: {len(pred_df):,}")

            if 'Failure_Probability' in pred_df.columns:
                mean_prob = pred_df['Failure_Probability'].mean()
                log_print(f"  Mean Failure Probability: {mean_prob:.2%}")

            if 'Risk_Level' in pred_df.columns:
                risk_dist = pred_df['Risk_Level'].value_counts()
                for risk, count in risk_dist.items():
                    pct = count / len(pred_df) * 100
                    log_print(f"    {risk}: {count:,} ({pct:.1f}%)")

        found_any_predictions = True

# Visualizations
if found_any_predictions:
    log_print("\n--- Creating Prediction Visualizations ---")

    # Priority: Visualize risk assessment if available, otherwise survival analysis, otherwise legacy
    viz_created = False

    # Try risk assessment first (most comprehensive)
    risk_12m_path = results_dir / 'risk_assessment_12M.csv'
    if risk_12m_path.exists():
        risk_df = pd.read_csv(risk_12m_path)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Risk Score distribution
        if 'Risk_Score' in risk_df.columns:
            ax = axes[0, 0]
            ax.hist(risk_df['Risk_Score'], bins=50, edgecolor='black', alpha=0.7, color='#e74c3c')
            ax.set_xlabel('Risk Score (0-100)', fontsize=11)
            ax.set_ylabel('Number of Equipment', fontsize=11)
            ax.set_title('Risk Score Distribution (12M)', fontsize=13, fontweight='bold')
            ax.axvline(risk_df['Risk_Score'].mean(), color='blue', linestyle='--',
                      label=f'Mean: {risk_df["Risk_Score"].mean():.1f}')
            ax.axvline(70, color='red', linestyle='--', linewidth=2, label='High Risk Threshold')
            ax.legend()
            ax.grid(alpha=0.3)

        # 2. Risk Category distribution
        if 'Risk_Category' in risk_df.columns:
            ax = axes[0, 1]
            risk_cat_dist = risk_df['Risk_Category'].value_counts()
            colors_cat = {'DÜŞÜK': '#2ecc71', 'ORTA': '#f39c12', 'YÜKSEK': '#e74c3c', 'KRİTİK': '#c0392b'}
            plot_colors = [colors_cat.get(cat, 'gray') for cat in risk_cat_dist.index]

            ax.bar(risk_cat_dist.index, risk_cat_dist.values, color=plot_colors, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Risk Category', fontsize=11)
            ax.set_ylabel('Number of Equipment', fontsize=11)
            ax.set_title('Risk Category Distribution (12M)', fontsize=13, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

            # Add value labels
            for i, (cat, count) in enumerate(risk_cat_dist.items()):
                pct = count / len(risk_df) * 100
                ax.text(i, count + max(risk_cat_dist.values)*0.02, f'{count:,}\n({pct:.1f}%)',
                       ha='center', va='bottom', fontweight='bold', fontsize=9)

        # 3. PoF vs CoF scatter
        if 'PoF_Probability' in risk_df.columns and 'CoF_Score' in risk_df.columns:
            ax = axes[1, 0]
            scatter = ax.scatter(risk_df['PoF_Probability'] * 100, risk_df['CoF_Score'],
                               c=risk_df['Risk_Score'], cmap='RdYlGn_r',
                               s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
            ax.set_xlabel('Probability of Failure (%)', fontsize=11)
            ax.set_ylabel('Consequence of Failure Score', fontsize=11)
            ax.set_title('Risk Matrix: PoF vs CoF (12M)', fontsize=13, fontweight='bold')
            ax.grid(alpha=0.3)

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Risk Score', rotation=270, labelpad=15)

            # Add quadrant lines
            ax.axhline(50, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(50, color='gray', linestyle='--', alpha=0.5)

        # 4. Risk by equipment class
        if 'Ekipman_Sınıfı' in risk_df.columns and 'Risk_Score' in risk_df.columns:
            ax = axes[1, 1]

            # Get top equipment classes
            top_classes = risk_df['Ekipman_Sınıfı'].value_counts().head(10).index
            class_risk = risk_df[risk_df['Ekipman_Sınıfı'].isin(top_classes)].groupby('Ekipman_Sınıfı')['Risk_Score'].mean()
            class_risk = class_risk.sort_values(ascending=False)

            ax.barh(range(len(class_risk)), class_risk.values, color='#e74c3c', alpha=0.7, edgecolor='black')
            ax.set_yticks(range(len(class_risk)))
            ax.set_yticklabels(class_risk.index, fontsize=9)
            ax.set_xlabel('Average Risk Score', fontsize=11)
            ax.set_ylabel('Equipment Class', fontsize=11)
            ax.set_title('Average Risk by Equipment Class (Top 10)', fontsize=12, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            ax.axvline(70, color='red', linestyle='--', linewidth=2, alpha=0.7, label='High Risk')
            ax.legend()

            # Add value labels
            for i, val in enumerate(class_risk.values):
                ax.text(val + 1, i, f'{val:.1f}', va='center', fontsize=8)

        plt.tight_layout()
        plt.savefig('outputs/eda/10_model_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
        log_print("\n✓ Saved: outputs/eda/10_model_predictions.png")
        viz_created = True

    # Fallback to survival analysis visualization
    elif pof_multi_horizon_path.exists() and not viz_created:
        pred_df = pd.read_csv(pof_multi_horizon_path)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot PoF distributions for each horizon
        horizons_to_plot = ['3M', '12M', '24M']
        ax_idx = 0

        for horizon in horizons_to_plot:
            pof_col = f'PoF_Probability_{horizon}'
            if pof_col in pred_df.columns and ax_idx < 3:
                ax = axes.flat[ax_idx]
                ax.hist(pred_df[pof_col] * 100, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
                ax.set_xlabel(f'PoF Probability ({horizon}) %', fontsize=11)
                ax.set_ylabel('Number of Equipment', fontsize=11)
                ax.set_title(f'PoF Distribution - {horizon} Horizon', fontsize=12, fontweight='bold')
                ax.axvline(pred_df[pof_col].mean() * 100, color='red', linestyle='--',
                          label=f'Mean: {pred_df[pof_col].mean():.2%}')
                ax.legend()
                ax.grid(alpha=0.3)
                ax_idx += 1

        # Risk category if available
        if 'Risk_Category' in pred_df.columns:
            ax = axes.flat[3]
            risk_dist = pred_df['Risk_Category'].value_counts()
            colors_cat = {'DÜŞÜK': 'green', 'ORTA': 'yellow', 'YÜKSEK': 'orange'}
            plot_colors = [colors_cat.get(cat, 'gray') for cat in risk_dist.index]

            ax.bar(risk_dist.index, risk_dist.values, color=plot_colors, alpha=0.7)
            ax.set_xlabel('Risk Category', fontsize=11)
            ax.set_ylabel('Number of Equipment', fontsize=11)
            ax.set_title('Risk Category Distribution', fontsize=13, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig('outputs/eda/10_model_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
        log_print("\n✓ Saved: outputs/eda/10_model_predictions.png")
        viz_created = True

if not found_any_predictions:
    log_print("\n⚠️  No prediction files found.")
    log_print("Run one of the following to generate predictions:")
    log_print("  • 06_model_training.py (Model 2: Chronic repeater classifier)")
    log_print("  • 09_survival_analysis.py (Model 1: Temporal PoF predictor)")
    log_print("  • 11_consequence_of_failure.py (Risk = PoF × CoF)")

# ============================================================================
# STEP 11: VOLTAGE-LEVEL ANALYSIS (NEW - From Step 9B)
# ============================================================================
log_print("\n" + "="*100)
log_print("STEP 11: VOLTAGE-LEVEL ANALYSIS")
log_print("="*100)

if 'Voltage_Class' in df.columns and df['Voltage_Class'].notna().any():
    log_print("\n--- Voltage Class Distribution ---")
    voltage_dist = df['Voltage_Class'].value_counts()
    log_print(f"\nTotal Voltage Classes: {len(voltage_dist)}")
    for v_class, count in voltage_dist.items():
        if pd.notna(v_class):
            pct = count / len(df) * 100
            log_print(f"  {v_class}: {count:,} equipment ({pct:.1f}%)")

    # Voltage-level failure patterns
    log_print("\n--- Failure Patterns by Voltage Level ---")
    for v_class in ['AG', 'OG', 'YG']:
        mask = df['Voltage_Class'] == v_class
        if mask.sum() > 0:
            avg_age = df.loc[mask, 'Ekipman_Yaşı_Yıl'].mean() if 'Ekipman_Yaşı_Yıl' in df.columns else 0
            avg_faults_12m = df.loc[mask, 'Arıza_Sayısı_12ay'].mean() if 'Arıza_Sayısı_12ay' in df.columns else 0
            recurring_pct = df.loc[mask, 'Tekrarlayan_Arıza_90gün_Flag'].sum() / mask.sum() * 100 if 'Tekrarlayan_Arıza_90gün_Flag' in df.columns else 0
            risk_score = df.loc[mask, 'Composite_PoF_Risk_Score'].mean() if 'Composite_PoF_Risk_Score' in df.columns else 0

            log_print(f"\n  {v_class}:")
            log_print(f"    Equipment count: {mask.sum():,}")
            log_print(f"    Avg age: {avg_age:.1f} years")
            log_print(f"    Avg 12M failures: {avg_faults_12m:.2f}")
            log_print(f"    Recurring faults: {recurring_pct:.1f}%")
            log_print(f"    Avg PoF risk score: {risk_score:.1f}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Failure rate by voltage
    if 'Arıza_Sayısı_12ay' in df.columns:
        ax = axes[0, 0]
        voltage_failures = df.groupby('Voltage_Class')['Arıza_Sayısı_12ay'].mean().sort_values(ascending=False)
        colors = ['#e74c3c' if v == 'OG' else '#3498db' if v == 'AG' else '#95a5a6' for v in voltage_failures.index]
        ax.bar(voltage_failures.index, voltage_failures.values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Voltage Class', fontsize=11)
        ax.set_ylabel('Average 12M Failures', fontsize=11)
        ax.set_title('Average Failure Rate by Voltage Level', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for i, (idx, val) in enumerate(voltage_failures.items()):
            ax.text(i, val + 0.02, f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

    # 2. Age distribution by voltage
    if 'Ekipman_Yaşı_Yıl' in df.columns:
        ax = axes[0, 1]
        voltage_classes = df['Voltage_Class'].dropna().unique()
        age_data = [df[df['Voltage_Class'] == vc]['Ekipman_Yaşı_Yıl'].dropna() for vc in voltage_classes if pd.notna(vc)]
        bp = ax.boxplot(age_data, labels=[vc for vc in voltage_classes if pd.notna(vc)], patch_artist=True)

        # Color boxes
        for patch, vc in zip(bp['boxes'], voltage_classes):
            if vc == 'OG':
                patch.set_facecolor('#e74c3c')
            elif vc == 'AG':
                patch.set_facecolor('#3498db')
            else:
                patch.set_facecolor('#95a5a6')
            patch.set_alpha(0.7)

        ax.set_xlabel('Voltage Class', fontsize=11)
        ax.set_ylabel('Equipment Age (Years)', fontsize=11)
        ax.set_title('Age Distribution by Voltage Level', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    # 3. Recurring faults by voltage
    if 'Tekrarlayan_Arıza_90gün_Flag' in df.columns:
        ax = axes[1, 0]
        recurring_by_voltage = df.groupby('Voltage_Class')['Tekrarlayan_Arıza_90gün_Flag'].apply(
            lambda x: x.sum() / len(x) * 100 if len(x) > 0 else 0
        ).sort_values(ascending=False)
        colors = ['#e74c3c' if v == 'OG' else '#3498db' if v == 'AG' else '#95a5a6' for v in recurring_by_voltage.index]
        ax.bar(recurring_by_voltage.index, recurring_by_voltage.values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Voltage Class', fontsize=11)
        ax.set_ylabel('Recurring Fault Rate (%)', fontsize=11)
        ax.set_title('Recurring Faults (90-day) by Voltage Level', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for i, (idx, val) in enumerate(recurring_by_voltage.items()):
            ax.text(i, val + 0.5, f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

    # 4. Risk score by voltage
    if 'Composite_PoF_Risk_Score' in df.columns:
        ax = axes[1, 1]
        voltage_classes = df['Voltage_Class'].dropna().unique()
        risk_data = [df[df['Voltage_Class'] == vc]['Composite_PoF_Risk_Score'].dropna() for vc in voltage_classes if pd.notna(vc)]

        parts = ax.violinplot(risk_data, positions=range(len(voltage_classes)), showmeans=True, showmedians=True)
        for pc, vc in zip(parts['bodies'], voltage_classes):
            if vc == 'OG':
                pc.set_facecolor('#e74c3c')
            elif vc == 'AG':
                pc.set_facecolor('#3498db')
            else:
                pc.set_facecolor('#95a5a6')
            pc.set_alpha(0.7)

        ax.set_xticks(range(len(voltage_classes)))
        ax.set_xticklabels([vc for vc in voltage_classes if pd.notna(vc)])
        ax.set_xlabel('Voltage Class', fontsize=11)
        ax.set_ylabel('PoF Risk Score', fontsize=11)
        ax.set_title('Risk Score Distribution by Voltage Level', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/eda/11_voltage_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    log_print("\n✓ Saved: outputs/eda/11_voltage_analysis.png")
else:
    log_print("\n⚠️  Voltage_Class column not found or empty")

# ============================================================================
# STEP 12: URBAN/RURAL ANALYSIS (NEW - From Step 9B)
# ============================================================================
log_print("\n" + "="*100)
log_print("STEP 12: URBAN/RURAL (GEOGRAPHIC TYPE) ANALYSIS")
log_print("="*100)

if 'Bölge_Tipi' in df.columns and df['Bölge_Tipi'].notna().any():
    log_print("\n--- Geographic Type Distribution ---")
    region_dist = df['Bölge_Tipi'].value_counts()
    log_print(f"\nTotal Regions: {len(region_dist)}")
    for region, count in region_dist.items():
        if pd.notna(region):
            pct = count / len(df) * 100
            log_print(f"  {region}: {count:,} equipment ({pct:.1f}%)")

    # Regional failure patterns
    log_print("\n--- Failure Patterns by Region ---")
    for region in ['Kentsel', 'Kırsal']:
        mask = df['Bölge_Tipi'] == region
        if mask.sum() > 0:
            avg_age = df.loc[mask, 'Ekipman_Yaşı_Yıl'].mean() if 'Ekipman_Yaşı_Yıl' in df.columns else 0
            avg_faults_12m = df.loc[mask, 'Arıza_Sayısı_12ay'].mean() if 'Arıza_Sayısı_12ay' in df.columns else 0
            avg_customers = df.loc[mask, 'total_customer_count_Avg'].mean() if 'total_customer_count_Avg' in df.columns else 0
            risk_score = df.loc[mask, 'Composite_PoF_Risk_Score'].mean() if 'Composite_PoF_Risk_Score' in df.columns else 0

            log_print(f"\n  {region}:")
            log_print(f"    Equipment count: {mask.sum():,}")
            log_print(f"    Avg age: {avg_age:.1f} years")
            log_print(f"    Avg 12M failures: {avg_faults_12m:.2f}")
            log_print(f"    Avg customers affected: {avg_customers:.1f}")
            log_print(f"    Avg PoF risk score: {risk_score:.1f}")

    # District breakdown
    if 'İlçe' in df.columns:
        log_print("\n--- District Breakdown ---")
        district_dist = df['İlçe'].value_counts().head(10)
        for district, count in district_dist.items():
            region = df[df['İlçe'] == district]['Bölge_Tipi'].mode()[0] if len(df[df['İlçe'] == district]) > 0 else 'Unknown'
            pct = count / len(df) * 100
            log_print(f"  {district}: {count:,} equipment ({pct:.1f}%) - {region}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Failure rate by region
    if 'Arıza_Sayısı_12ay' in df.columns:
        ax = axes[0, 0]
        region_failures = df.groupby('Bölge_Tipi')['Arıza_Sayısı_12ay'].mean().sort_values(ascending=False)
        colors = ['#e67e22' if r == 'Kırsal' else '#27ae60' for r in region_failures.index]
        ax.bar(region_failures.index, region_failures.values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Region Type', fontsize=11)
        ax.set_ylabel('Average 12M Failures', fontsize=11)
        ax.set_title('Average Failure Rate by Region', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for i, (idx, val) in enumerate(region_failures.items()):
            ax.text(i, val + 0.02, f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

    # 2. Age distribution by region
    if 'Ekipman_Yaşı_Yıl' in df.columns:
        ax = axes[0, 1]
        regions = df['Bölge_Tipi'].dropna().unique()
        age_data = [df[df['Bölge_Tipi'] == r]['Ekipman_Yaşı_Yıl'].dropna() for r in regions if pd.notna(r)]
        bp = ax.boxplot(age_data, labels=[r for r in regions if pd.notna(r)], patch_artist=True)

        # Color boxes
        for patch, r in zip(bp['boxes'], regions):
            if r == 'Kırsal':
                patch.set_facecolor('#e67e22')
            elif r == 'Kentsel':
                patch.set_facecolor('#27ae60')
            patch.set_alpha(0.7)

        ax.set_xlabel('Region Type', fontsize=11)
        ax.set_ylabel('Equipment Age (Years)', fontsize=11)
        ax.set_title('Age Distribution by Region', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    # 3. Customer impact by region
    if 'total_customer_count_Avg' in df.columns:
        ax = axes[1, 0]
        region_customers = df.groupby('Bölge_Tipi')['total_customer_count_Avg'].mean().sort_values(ascending=False)
        colors = ['#e67e22' if r == 'Kırsal' else '#27ae60' for r in region_customers.index]
        ax.bar(region_customers.index, region_customers.values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Region Type', fontsize=11)
        ax.set_ylabel('Average Customers Affected', fontsize=11)
        ax.set_title('Customer Impact by Region', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for i, (idx, val) in enumerate(region_customers.items()):
            ax.text(i, val + 2, f'{val:.0f}', ha='center', va='bottom', fontweight='bold')

    # 4. Equipment count by district
    if 'İlçe' in df.columns:
        ax = axes[1, 1]
        district_counts = df['İlçe'].value_counts().head(10)
        colors = []
        for district in district_counts.index:
            region = df[df['İlçe'] == district]['Bölge_Tipi'].mode()[0] if len(df[df['İlçe'] == district]) > 0 else 'Unknown'
            colors.append('#e67e22' if region == 'Kırsal' else '#27ae60')

        ax.barh(range(len(district_counts)), district_counts.values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(district_counts)))
        ax.set_yticklabels(district_counts.index)
        ax.set_xlabel('Equipment Count', fontsize=11)
        ax.set_ylabel('District', fontsize=11)
        ax.set_title('Top 10 Districts by Equipment Count', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, val in enumerate(district_counts.values):
            ax.text(val + 5, i, f'{val:,}', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('outputs/eda/12_urban_rural_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    log_print("\n✓ Saved: outputs/eda/12_urban_rural_analysis.png")
else:
    log_print("\n⚠️  Bölge_Tipi column not found or empty")

# ============================================================================
# STEP 13: SEASONAL ANALYSIS (NEW - From Step 9B)
# ============================================================================
log_print("\n" + "="*100)
log_print("STEP 13: SEASONAL FAILURE PATTERN ANALYSIS")
log_print("="*100)

if 'Son_Arıza_Mevsim' in df.columns and df['Son_Arıza_Mevsim'].notna().any():
    log_print("\n--- Seasonal Distribution of Last Faults ---")
    season_dist = df['Son_Arıza_Mevsim'].value_counts()
    total_with_season = df['Son_Arıza_Mevsim'].notna().sum()

    log_print(f"\nEquipment with seasonal data: {total_with_season:,} ({total_with_season/len(df)*100:.1f}%)")
    log_print(f"\nSeason breakdown:")
    for season in ['Yaz', 'Kış', 'İlkbahar', 'Sonbahar']:
        count = season_dist.get(season, 0)
        if count > 0:
            pct = count / total_with_season * 100
            log_print(f"  {season}: {count:,} equipment ({pct:.1f}%)")

    # Seasonal failure analysis
    log_print("\n--- Failure Characteristics by Season ---")
    for season in ['Yaz', 'Kış', 'İlkbahar', 'Sonbahar']:
        mask = df['Son_Arıza_Mevsim'] == season
        if mask.sum() > 0:
            avg_faults = df.loc[mask, 'Toplam_Arıza_Sayisi_Lifetime'].mean() if 'Toplam_Arıza_Sayisi_Lifetime' in df.columns else 0
            recurring_pct = df.loc[mask, 'Tekrarlayan_Arıza_90gün_Flag'].sum() / mask.sum() * 100 if 'Tekrarlayan_Arıza_90gün_Flag' in df.columns else 0

            log_print(f"\n  {season}:")
            log_print(f"    Equipment count: {mask.sum():,}")
            log_print(f"    Avg lifetime failures: {avg_faults:.2f}")
            log_print(f"    Recurring fault rate: {recurring_pct:.1f}%")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Seasonal distribution (pie chart)
    ax = axes[0, 0]
    colors_season = {'Yaz': '#e74c3c', 'Kış': '#3498db', 'İlkbahar': '#2ecc71', 'Sonbahar': '#f39c12'}
    season_colors = [colors_season.get(s, '#95a5a6') for s in season_dist.index]
    wedges, texts, autotexts = ax.pie(season_dist.values, labels=season_dist.index, autopct='%1.1f%%',
                                        colors=season_colors, startangle=90)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax.set_title('Last Fault Distribution by Season', fontsize=13, fontweight='bold')

    # 2. Seasonal failure counts (bar chart)
    if 'Toplam_Arıza_Sayisi_Lifetime' in df.columns:
        ax = axes[0, 1]
        season_failures = df.groupby('Son_Arıza_Mevsim')['Toplam_Arıza_Sayisi_Lifetime'].mean()
        season_order = ['Yaz', 'Kış', 'İlkbahar', 'Sonbahar']
        season_failures = season_failures.reindex([s for s in season_order if s in season_failures.index])
        colors = [colors_season.get(s, '#95a5a6') for s in season_failures.index]

        ax.bar(season_failures.index, season_failures.values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Season', fontsize=11)
        ax.set_ylabel('Average Lifetime Failures', fontsize=11)
        ax.set_title('Average Failure Count by Season', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for i, (idx, val) in enumerate(season_failures.items()):
            ax.text(i, val + 0.05, f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

    # 3. Equipment class × Season heatmap
    if 'Equipment_Class_Primary' in df.columns:
        ax = axes[1, 0]

        # Get top 8 equipment classes
        top_classes = df['Equipment_Class_Primary'].value_counts().head(8).index
        df_top = df[df['Equipment_Class_Primary'].isin(top_classes)]

        # Create crosstab
        season_class_ct = pd.crosstab(df_top['Equipment_Class_Primary'], df_top['Son_Arıza_Mevsim'])
        season_order = ['Yaz', 'Kış', 'İlkbahar', 'Sonbahar']
        season_class_ct = season_class_ct[[s for s in season_order if s in season_class_ct.columns]]

        # Normalize by row (equipment class)
        season_class_pct = season_class_ct.div(season_class_ct.sum(axis=1), axis=0) * 100

        im = ax.imshow(season_class_pct.values, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(season_class_pct.columns)))
        ax.set_yticks(range(len(season_class_pct.index)))
        ax.set_xticklabels(season_class_pct.columns)
        ax.set_yticklabels(season_class_pct.index)
        ax.set_xlabel('Season', fontsize=11)
        ax.set_ylabel('Equipment Class', fontsize=11)
        ax.set_title('Equipment Class × Season Heatmap (%)', fontsize=13, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Percentage', rotation=270, labelpad=15)

        # Add text annotations
        for i in range(len(season_class_pct.index)):
            for j in range(len(season_class_pct.columns)):
                text = ax.text(j, i, f'{season_class_pct.iloc[i, j]:.0f}',
                              ha="center", va="center", color="black", fontsize=8)

    # 4. Recurring faults by season
    if 'Tekrarlayan_Arıza_90gün_Flag' in df.columns:
        ax = axes[1, 1]
        recurring_by_season = df.groupby('Son_Arıza_Mevsim')['Tekrarlayan_Arıza_90gün_Flag'].apply(
            lambda x: x.sum() / len(x) * 100 if len(x) > 0 else 0
        )
        season_order = ['Yaz', 'Kış', 'İlkbahar', 'Sonbahar']
        recurring_by_season = recurring_by_season.reindex([s for s in season_order if s in recurring_by_season.index])
        colors = [colors_season.get(s, '#95a5a6') for s in recurring_by_season.index]

        ax.bar(recurring_by_season.index, recurring_by_season.values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Season', fontsize=11)
        ax.set_ylabel('Recurring Fault Rate (%)', fontsize=11)
        ax.set_title('Recurring Faults (90-day) by Season', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for i, (idx, val) in enumerate(recurring_by_season.items()):
            ax.text(i, val + 0.3, f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('outputs/eda/13_seasonal_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    log_print("\n✓ Saved: outputs/eda/13_seasonal_analysis.png")
else:
    log_print("\n⚠️  Son_Arıza_Mevsim column not found or empty")

# ============================================================================
# STEP 14: CUSTOMER RATIOS & LOADING ANALYSIS (NEW - From Step 9B)
# ============================================================================
log_print("\n" + "="*100)
log_print("STEP 14: CUSTOMER RATIOS & LOADING INTENSITY ANALYSIS")
log_print("="*100)

# Customer ratios analysis
customer_ratio_cols = ['Kentsel_Müşteri_Oranı', 'Kırsal_Müşteri_Oranı', 'OG_Müşteri_Oranı']
has_customer_ratios = any(col in df.columns for col in customer_ratio_cols)

if has_customer_ratios:
    log_print("\n--- Customer Type Ratios Statistics ---")
    for col in customer_ratio_cols:
        if col in df.columns:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                log_print(f"\n{col}:")
                log_print(f"  Mean: {col_data.mean():.2%}")
                log_print(f"  Median: {col_data.median():.2%}")
                log_print(f"  Min: {col_data.min():.2%}")
                log_print(f"  Max: {col_data.max():.2%}")
                log_print(f"  Std Dev: {col_data.std():.2%}")

# Loading intensity analysis
if 'Ekipman_Yoğunluk_Skoru' in df.columns:
    log_print("\n--- Equipment Loading Score Statistics ---")
    loading_data = df['Ekipman_Yoğunluk_Skoru'].replace([np.inf, -np.inf], np.nan).dropna()
    if len(loading_data) > 0:
        log_print(f"  Mean: {loading_data.mean():.4f}")
        log_print(f"  Median: {loading_data.median():.4f}")
        log_print(f"  Min: {loading_data.min():.4f}")
        log_print(f"  Max: {loading_data.max():.4f}")
        log_print(f"  95th percentile: {loading_data.quantile(0.95):.4f}")

# Customer-weighted risk
if 'Müşteri_Başına_Risk' in df.columns:
    log_print("\n--- Customer-Weighted Risk Statistics ---")
    risk_data = df['Müşteri_Başına_Risk'].dropna()
    if len(risk_data) > 0:
        log_print(f"  Mean: {risk_data.mean():.3f}")
        log_print(f"  Median: {risk_data.median():.3f}")
        log_print(f"  Min: {risk_data.min():.3f}")
        log_print(f"  Max: {risk_data.max():.3f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Customer ratios distribution
ax = axes[0, 0]
customer_ratio_data = []
customer_ratio_labels = []
for col in customer_ratio_cols:
    if col in df.columns:
        data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) > 0:
            customer_ratio_data.append(data)
            customer_ratio_labels.append(col.replace('_', ' '))

if customer_ratio_data:
    bp = ax.boxplot(customer_ratio_data, labels=customer_ratio_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#3498db')
        patch.set_alpha(0.7)
    ax.set_ylabel('Ratio', fontsize=11)
    ax.set_title('Customer Type Ratio Distributions', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=15)

# 2. Loading score distribution
if 'Ekipman_Yoğunluk_Skoru' in df.columns:
    ax = axes[0, 1]
    loading_data = df['Ekipman_Yoğunluk_Skoru'].replace([np.inf, -np.inf], np.nan).dropna()
    if len(loading_data) > 0:
        # Cap at 95th percentile for visualization
        p95 = loading_data.quantile(0.95)
        loading_capped = loading_data[loading_data <= p95]

        ax.hist(loading_capped, bins=50, edgecolor='black', alpha=0.7, color='#e74c3c')
        ax.set_xlabel('Loading Score (Recency-Based)', fontsize=11)
        ax.set_ylabel('Number of Equipment', fontsize=11)
        ax.set_title('Equipment Loading Score Distribution (capped at 95th percentile)', fontsize=13, fontweight='bold')
        ax.axvline(loading_capped.median(), color='blue', linestyle='--', label=f'Median: {loading_capped.median():.4f}')
        ax.axvline(loading_capped.mean(), color='red', linestyle='--', label=f'Mean: {loading_capped.mean():.4f}')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

# 3. Customer-weighted risk
if 'Müşteri_Başına_Risk' in df.columns:
    ax = axes[1, 0]
    risk_data = df['Müşteri_Başına_Risk'].dropna()
    if len(risk_data) > 0:
        ax.hist(risk_data, bins=50, edgecolor='black', alpha=0.7, color='#9b59b6')
        ax.set_xlabel('Customer-Weighted Risk', fontsize=11)
        ax.set_ylabel('Number of Equipment', fontsize=11)
        ax.set_title('Customer-Weighted Risk Distribution', fontsize=13, fontweight='bold')
        ax.axvline(risk_data.median(), color='blue', linestyle='--', label=f'Median: {risk_data.median():.2f}')
        ax.axvline(risk_data.mean(), color='red', linestyle='--', label=f'Mean: {risk_data.mean():.2f}')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

# 4. Correlation: Loading score vs failures
if 'Ekipman_Yoğunluk_Skoru' in df.columns and 'Arıza_Sayısı_12ay' in df.columns:
    ax = axes[1, 1]

    # Prepare data
    plot_df = df[['Ekipman_Yoğunluk_Skoru', 'Arıza_Sayısı_12ay']].copy()
    plot_df['Ekipman_Yoğunluk_Skoru'] = plot_df['Ekipman_Yoğunluk_Skoru'].replace([np.inf, -np.inf], np.nan)
    plot_df = plot_df.dropna()

    if len(plot_df) > 0:
        # Cap loading score at 95th percentile for visualization
        p95 = plot_df['Ekipman_Yoğunluk_Skoru'].quantile(0.95)
        plot_df_capped = plot_df[plot_df['Ekipman_Yoğunluk_Skoru'] <= p95]

        ax.scatter(plot_df_capped['Ekipman_Yoğunluk_Skoru'], plot_df_capped['Arıza_Sayısı_12ay'],
                  alpha=0.5, s=30, color='#e74c3c')

        # Add trend line
        z = np.polyfit(plot_df_capped['Ekipman_Yoğunluk_Skoru'], plot_df_capped['Arıza_Sayısı_12ay'], 1)
        p = np.poly1d(z)
        ax.plot(plot_df_capped['Ekipman_Yoğunluk_Skoru'],
               p(plot_df_capped['Ekipman_Yoğunluk_Skoru']),
               "r--", linewidth=2, label='Trend')

        # Calculate correlation
        corr = plot_df_capped['Ekipman_Yoğunluk_Skoru'].corr(plot_df_capped['Arıza_Sayısı_12ay'])

        ax.set_xlabel('Loading Score (Recency-Based)', fontsize=11)
        ax.set_ylabel('12M Failures', fontsize=11)
        ax.set_title(f'Loading Score vs Failures (r={corr:.3f})', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/eda/14_customer_loading_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
log_print("\n✓ Saved: outputs/eda/14_customer_loading_analysis.png")

# ============================================================================
# STEP 15: CAUSE CODE ANALYSIS (MODULE 3 REQUIREMENT)
# ============================================================================
log_print("\n" + "="*100)
log_print("STEP 15: CAUSE CODE ANALYSIS (MODULE 3)")
log_print("="*100)

# Check for cause code columns
cause_code_cols = {
    'Arıza_Nedeni_İlk': 'First Cause',
    'Arıza_Nedeni_Son': 'Last Cause',
    'Arıza_Nedeni_Sık': 'Most Common Cause',
    'Arıza_Nedeni_Çeşitlilik': 'Cause Diversity',
    'Arıza_Nedeni_Tutarlılık': 'Cause Consistency',
    'Tek_Neden_Flag': 'Single Dominant Cause Flag',
    'Çok_Nedenli_Flag': 'Multiple Causes Flag',
    'Neden_Değişim_Flag': 'Cause Changed Flag',
    'Ekipman_Neden_Risk_Skoru': 'Equipment×Cause Risk Score'
}

available_cause_cols = {k: v for k, v in cause_code_cols.items() if k in df.columns}

if available_cause_cols:
    log_print(f"\n✓ Found {len(available_cause_cols)} cause code features:")
    for col, desc in available_cause_cols.items():
        log_print(f"  • {col}: {desc}")

    # Cause code statistics
    log_print("\n--- Cause Code Statistics ---")

    # Cause diversity
    if 'Arıza_Nedeni_Çeşitlilik' in df.columns:
        diversity_stats = df['Arıza_Nedeni_Çeşitlilik'].describe()
        log_print(f"\nCause Diversity (types per equipment):")
        log_print(f"  Mean: {diversity_stats['mean']:.2f} types")
        log_print(f"  Median: {diversity_stats['50%']:.0f} types")
        log_print(f"  Max: {diversity_stats['max']:.0f} types")

    # Cause consistency
    if 'Arıza_Nedeni_Tutarlılık' in df.columns:
        consistency_stats = df['Arıza_Nedeni_Tutarlılık'].describe()
        log_print(f"\nCause Consistency (% of failures with most common cause):")
        log_print(f"  Mean: {consistency_stats['mean']:.2%}")
        log_print(f"  Median: {consistency_stats['50%']:.2%}")
        log_print(f"  Equipment with 100% consistency: {(df['Arıza_Nedeni_Tutarlılık'] == 1.0).sum():,}")

    # Single dominant cause flag
    if 'Tek_Neden_Flag' in df.columns:
        tek_neden_count = df['Tek_Neden_Flag'].sum()
        tek_neden_pct = tek_neden_count / len(df) * 100
        log_print(f"\nSingle Dominant Cause (≥80% consistency):")
        log_print(f"  Equipment count: {tek_neden_count:,} ({tek_neden_pct:.1f}%)")

        # Impact on failures
        if 'Arıza_Sayısı_12ay' in df.columns:
            tek_neden_failures = df[df['Tek_Neden_Flag'] == 1]['Arıza_Sayısı_12ay'].mean()
            multi_neden_failures = df[df['Tek_Neden_Flag'] == 0]['Arıza_Sayısı_12ay'].mean()
            log_print(f"  Avg 12M failures (single cause): {tek_neden_failures:.2f}")
            log_print(f"  Avg 12M failures (multiple causes): {multi_neden_failures:.2f}")

    # Multiple causes flag
    if 'Çok_Nedenli_Flag' in df.columns:
        multi_count = df['Çok_Nedenli_Flag'].sum()
        multi_pct = multi_count / len(df) * 100
        log_print(f"\nMultiple Causes (≥3 different types):")
        log_print(f"  Equipment count: {multi_count:,} ({multi_pct:.1f}%)")

    # Cause evolution
    if 'Neden_Değişim_Flag' in df.columns:
        changed_count = df['Neden_Değişim_Flag'].sum()
        changed_pct = changed_count / len(df) * 100
        log_print(f"\nCause Evolution (first ≠ last):")
        log_print(f"  Equipment count: {changed_count:,} ({changed_pct:.1f}%)")

    # Most common causes
    if 'Arıza_Nedeni_Sık' in df.columns:
        log_print(f"\n--- Most Common Cause Codes ---")
        cause_dist = df['Arıza_Nedeni_Sık'].value_counts().head(15)
        log_print(f"\nTop 15 Cause Codes:")
        for i, (cause, count) in enumerate(cause_dist.items(), 1):
            pct = count / df['Arıza_Nedeni_Sık'].notna().sum() * 100
            log_print(f"  {i:2d}. {cause}: {count:,} ({pct:.1f}%)")

    # Equipment Class × Cause Code analysis
    if 'Equipment_Class_Primary' in df.columns and 'Arıza_Nedeni_Sık' in df.columns:
        log_print(f"\n--- Equipment Class × Cause Code Patterns ---")

        # Get top equipment classes and causes
        top_classes = df['Equipment_Class_Primary'].value_counts().head(8).index
        top_causes = df['Arıza_Nedeni_Sık'].value_counts().head(10).index

        # Create crosstab
        class_cause_ct = pd.crosstab(
            df[df['Equipment_Class_Primary'].isin(top_classes)]['Equipment_Class_Primary'],
            df[df['Arıza_Nedeni_Sık'].isin(top_causes)]['Arıza_Nedeni_Sık']
        )

        log_print(f"\nCrosstab created: {len(class_cause_ct)} classes × {len(class_cause_ct.columns)} causes")

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # 1. Cause consistency distribution
    if 'Arıza_Nedeni_Tutarlılık' in df.columns:
        ax = axes[0, 0]
        consistency_data = df['Arıza_Nedeni_Tutarlılık'].dropna()
        ax.hist(consistency_data, bins=50, edgecolor='black', alpha=0.7, color='#3498db')
        ax.set_xlabel('Cause Consistency (0=diverse, 1=single cause)', fontsize=11)
        ax.set_ylabel('Number of Equipment', fontsize=11)
        ax.set_title('Cause Consistency Distribution', fontsize=13, fontweight='bold')
        ax.axvline(0.8, color='red', linestyle='--', linewidth=2,
                   label=f'80% threshold (Tek_Neden_Flag)')
        ax.axvline(consistency_data.mean(), color='orange', linestyle='--',
                   label=f'Mean: {consistency_data.mean():.2%}')
        ax.legend()
        ax.grid(alpha=0.3)

    # 2. Top cause codes distribution
    if 'Arıza_Nedeni_Sık' in df.columns:
        ax = axes[0, 1]
        cause_dist = df['Arıza_Nedeni_Sık'].value_counts().head(15)
        ax.barh(range(len(cause_dist)), cause_dist.values, color='#e74c3c', alpha=0.7)
        ax.set_yticks(range(len(cause_dist)))
        ax.set_yticklabels(cause_dist.index, fontsize=9)
        ax.set_xlabel('Number of Equipment', fontsize=11)
        ax.set_ylabel('Cause Code', fontsize=11)
        ax.set_title('Top 15 Most Common Cause Codes', fontsize=13, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, val in enumerate(cause_dist.values):
            ax.text(val + 5, i, f'{val:,}', va='center', fontsize=8)

    # 3. Equipment Class × Cause Code heatmap
    if 'Equipment_Class_Primary' in df.columns and 'Arıza_Nedeni_Sık' in df.columns:
        ax = axes[1, 0]

        # Normalize by row (equipment class)
        class_cause_pct = class_cause_ct.div(class_cause_ct.sum(axis=1), axis=0) * 100

        # Create heatmap
        im = ax.imshow(class_cause_pct.values, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(class_cause_pct.columns)))
        ax.set_yticks(range(len(class_cause_pct.index)))
        ax.set_xticklabels(class_cause_pct.columns, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(class_cause_pct.index, fontsize=9)
        ax.set_xlabel('Cause Code', fontsize=11)
        ax.set_ylabel('Equipment Class', fontsize=11)
        ax.set_title('Equipment Class × Cause Code Heatmap (%)', fontsize=13, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Percentage', rotation=270, labelpad=15)

        # Add text annotations (only for larger cells)
        for i in range(len(class_cause_pct.index)):
            for j in range(len(class_cause_pct.columns)):
                val = class_cause_pct.iloc[i, j]
                if val > 5:  # Only show if > 5%
                    ax.text(j, i, f'{val:.0f}', ha="center", va="center",
                           color="black" if val < 50 else "white", fontsize=7)

    # 4. Tek_Neden_Flag impact on failures
    if 'Tek_Neden_Flag' in df.columns and 'Arıza_Sayısı_12ay' in df.columns:
        ax = axes[1, 1]

        # Box plot comparing single vs multiple cause equipment
        tek_neden_data = df[df['Tek_Neden_Flag'] == 1]['Arıza_Sayısı_12ay'].dropna()
        multi_neden_data = df[df['Tek_Neden_Flag'] == 0]['Arıza_Sayısı_12ay'].dropna()

        bp = ax.boxplot([tek_neden_data, multi_neden_data],
                        labels=['Single Dominant Cause', 'Multiple Causes'],
                        patch_artist=True)

        # Color boxes
        bp['boxes'][0].set_facecolor('#2ecc71')
        bp['boxes'][1].set_facecolor('#e67e22')
        for box in bp['boxes']:
            box.set_alpha(0.7)

        ax.set_ylabel('12M Failures', fontsize=11)
        ax.set_title('Failure Rate: Single vs Multiple Causes', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add mean markers
        means = [tek_neden_data.mean(), multi_neden_data.mean()]
        ax.scatter([1, 2], means, color='red', s=100, zorder=5, marker='D',
                  label='Mean', edgecolors='black', linewidth=1)
        ax.legend()

    plt.tight_layout()
    plt.savefig('outputs/eda/15_cause_code_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    log_print("\n✓ Saved: outputs/eda/15_cause_code_analysis.png")
else:
    log_print("\n⚠️  No cause code features found - run 02_data_transformation.py and 03_feature_engineering.py first")

# ============================================================================
# STEP 16: RECURRING FAILURE (CHRONIC REPEATER) ANALYSIS
# ============================================================================
log_print("\n" + "="*100)
log_print("STEP 16: RECURRING FAILURE (CHRONIC REPEATER) ANALYSIS")
log_print("="*100)

recurring_30d_col = 'Tekrarlayan_Arıza_30gün_Flag'
recurring_90d_col = 'Tekrarlayan_Arıza_90gün_Flag'

has_recurring = recurring_30d_col in df.columns or recurring_90d_col in df.columns

if has_recurring:
    log_print("\n--- Recurring Failure Statistics ---")

    # 30-day recurring
    if recurring_30d_col in df.columns:
        count_30d = df[recurring_30d_col].sum()
        pct_30d = count_30d / len(df) * 100
        log_print(f"\n30-Day Recurring Failures:")
        log_print(f"  Equipment count: {count_30d:,} ({pct_30d:.1f}%)")
        log_print(f"  Definition: At least 2 failures within 30 days")

    # 90-day recurring
    if recurring_90d_col in df.columns:
        count_90d = df[recurring_90d_col].sum()
        pct_90d = count_90d / len(df) * 100
        log_print(f"\n90-Day Recurring Failures:")
        log_print(f"  Equipment count: {count_90d:,} ({pct_90d:.1f}%)")
        log_print(f"  Definition: At least 2 failures within 90 days")

    # Recurring vs MTBF
    if recurring_90d_col in df.columns and 'MTBF_Gün' in df.columns:
        recurring_mtbf = df[df[recurring_90d_col] == 1]['MTBF_Gün'].mean()
        non_recurring_mtbf = df[df[recurring_90d_col] == 0]['MTBF_Gün'].mean()
        log_print(f"\nMTBF Comparison:")
        log_print(f"  Recurring equipment: {recurring_mtbf:.1f} days")
        log_print(f"  Non-recurring equipment: {non_recurring_mtbf:.1f} days")
        log_print(f"  Difference: {non_recurring_mtbf - recurring_mtbf:.1f} days")

    # Recurring by equipment class
    if recurring_90d_col in df.columns and 'Equipment_Class_Primary' in df.columns:
        log_print(f"\n--- Recurring Rate by Equipment Class ---")

        class_recurring = df.groupby('Equipment_Class_Primary')[recurring_90d_col].agg(['sum', 'count', 'mean'])
        class_recurring = class_recurring.sort_values('mean', ascending=False).head(10)

        log_print(f"\nTop 10 Classes by Recurring Rate:")
        for i, (cls, row) in enumerate(class_recurring.iterrows(), 1):
            recurring_pct = row['mean'] * 100
            log_print(f"  {i:2d}. {cls:<25} {row['sum']:>4.0f}/{row['count']:<4.0f} ({recurring_pct:>5.1f}%)")

    # Recurring by district
    if recurring_90d_col in df.columns and 'İlçe' in df.columns:
        log_print(f"\n--- Recurring Rate by District ---")

        district_recurring = df.groupby('İlçe')[recurring_90d_col].agg(['sum', 'count', 'mean'])
        district_recurring = district_recurring.sort_values('mean', ascending=False)

        log_print(f"\nAll Districts:")
        for district, row in district_recurring.iterrows():
            recurring_pct = row['mean'] * 100
            log_print(f"  {district:<20} {row['sum']:>4.0f}/{row['count']:<4.0f} ({recurring_pct:>5.1f}%)")

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 30-day vs 90-day comparison
    ax = axes[0, 0]
    recurring_data = []
    recurring_labels = []

    if recurring_30d_col in df.columns:
        count_30d = df[recurring_30d_col].sum()
        recurring_data.append(count_30d)
        recurring_labels.append(f'30-Day\n({count_30d:,})')

    if recurring_90d_col in df.columns:
        count_90d = df[recurring_90d_col].sum()
        recurring_data.append(count_90d)
        recurring_labels.append(f'90-Day\n({count_90d:,})')

    if recurring_data:
        colors = ['#e74c3c', '#e67e22'][:len(recurring_data)]
        ax.bar(recurring_labels, recurring_data, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Number of Equipment', fontsize=11)
        ax.set_title('Recurring Failure Equipment Count', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add percentage labels
        for i, (label, count) in enumerate(zip(recurring_labels, recurring_data)):
            pct = count / len(df) * 100
            ax.text(i, count + max(recurring_data)*0.02, f'{pct:.1f}%',
                   ha='center', va='bottom', fontweight='bold')

    # 2. Recurring rate by equipment class
    if recurring_90d_col in df.columns and 'Equipment_Class_Primary' in df.columns:
        ax = axes[0, 1]

        class_recurring_rate = df.groupby('Equipment_Class_Primary')[recurring_90d_col].mean() * 100
        class_recurring_rate = class_recurring_rate.sort_values(ascending=False).head(12)

        ax.barh(range(len(class_recurring_rate)), class_recurring_rate.values,
               color='#e74c3c', alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(class_recurring_rate)))
        ax.set_yticklabels(class_recurring_rate.index, fontsize=9)
        ax.set_xlabel('Recurring Failure Rate (%)', fontsize=11)
        ax.set_ylabel('Equipment Class', fontsize=11)
        ax.set_title('Recurring Rate by Equipment Class (Top 12)', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, val in enumerate(class_recurring_rate.values):
            ax.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=8)

    # 3. Geographic distribution (district)
    if recurring_90d_col in df.columns and 'İlçe' in df.columns:
        ax = axes[1, 0]

        district_recurring = df.groupby('İlçe')[recurring_90d_col].sum().sort_values(ascending=False)

        colors_dist = ['#e74c3c' if d == 'Salihli' else '#3498db' if d == 'Alaşehir' else '#2ecc71'
                      for d in district_recurring.index]

        ax.bar(district_recurring.index, district_recurring.values,
              color=colors_dist, alpha=0.7, edgecolor='black')
        ax.set_xlabel('District', fontsize=11)
        ax.set_ylabel('Recurring Equipment Count', fontsize=11)
        ax.set_title('Chronic Repeaters by District (90-day)', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for i, (district, count) in enumerate(district_recurring.items()):
            ax.text(i, count + max(district_recurring.values)*0.02, f'{count:.0f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=9)

    # 4. Recurring vs Non-Recurring MTBF comparison
    if recurring_90d_col in df.columns and 'MTBF_Gün' in df.columns:
        ax = axes[1, 1]

        recurring_mtbf_data = df[df[recurring_90d_col] == 1]['MTBF_Gün'].dropna()
        non_recurring_mtbf_data = df[df[recurring_90d_col] == 0]['MTBF_Gün'].dropna()

        # Filter outliers for better visualization
        recurring_mtbf_capped = recurring_mtbf_data[recurring_mtbf_data < recurring_mtbf_data.quantile(0.95)]
        non_recurring_mtbf_capped = non_recurring_mtbf_data[non_recurring_mtbf_data < non_recurring_mtbf_data.quantile(0.95)]

        bp = ax.boxplot([recurring_mtbf_capped, non_recurring_mtbf_capped],
                        labels=['Recurring\n(90-day)', 'Non-Recurring'],
                        patch_artist=True)

        # Color boxes
        bp['boxes'][0].set_facecolor('#e74c3c')
        bp['boxes'][1].set_facecolor('#2ecc71')
        for box in bp['boxes']:
            box.set_alpha(0.7)

        ax.set_ylabel('MTBF (Days)', fontsize=11)
        ax.set_title('MTBF: Recurring vs Non-Recurring Equipment', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add mean markers
        means = [recurring_mtbf_capped.mean(), non_recurring_mtbf_capped.mean()]
        ax.scatter([1, 2], means, color='blue', s=100, zorder=5, marker='D',
                  label=f'Mean', edgecolors='black', linewidth=1)
        ax.legend()

    plt.tight_layout()
    plt.savefig('outputs/eda/16_recurring_failure_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    log_print("\n✓ Saved: outputs/eda/16_recurring_failure_analysis.png")
else:
    log_print("\n⚠️  No recurring failure columns found - run 02_data_transformation.py first")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
log_print("\n" + "="*100)
log_print("EDA COMPLETE - SUMMARY")
log_print("="*100)

log_print(f"\n📊 ANALYSIS SUMMARY:")
log_print(f"   Total Equipment: {df.shape[0]:,}")
log_print(f"   Total Features: {df.shape[1]}")
if 'Equipment_Class_Primary' in df.columns:
    log_print(f"   Equipment Classes: {df['Equipment_Class_Primary'].nunique()}")
if 'Arıza_Sayısı_12ay' in df.columns:
    failure_rate = (df['Arıza_Sayısı_12ay'] > 0).sum() / len(df) * 100
    log_print(f"   12M Failure Rate: {failure_rate:.1f}%")

log_print(f"\n📂 OUTPUT FILES:")
visualizations = list(Path('outputs/eda').glob('*.png'))
log_print(f"   Visualizations: outputs/eda/ ({len(visualizations)} PNG files)")
log_print(f"   Summary Report: reports/eda_summary.txt")

log_print(f"\n✅ KEY INSIGHTS TO REVIEW:")
log_print(f"   1. Equipment class distribution and failure patterns")
log_print(f"   2. Failure rates across different time horizons")
log_print(f"   3. Equipment age vs failure relationship")
log_print(f"   4. Geographic cluster patterns")
log_print(f"   5. Risk category distribution")
log_print(f"   6. Feature correlations and relationships")
log_print(f"   7. Voltage-level failure patterns (MV vs LV)")
log_print(f"   8. Urban vs Rural infrastructure analysis")
log_print(f"   9. Seasonal failure patterns")
log_print(f"   10. Customer ratios and loading intensity")
log_print(f"   11. Cause code analysis and Equipment×Cause interactions [NEW]")
log_print(f"   12. Recurring failure (chronic repeater) patterns [NEW]")
log_print(f"   13. Model predictions (PoF, CoF, Risk assessment) [ENHANCED]")

log_print("\n" + "="*100)
log_print("EDA PIPELINE COMPLETE")
log_print("="*100)

# Close report file
report_file.close()
print(f"\n✅ Full report saved: reports/eda_summary.txt")