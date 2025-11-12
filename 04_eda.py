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

# Check if predictions exist
predictions_dir = Path('predictions')
if predictions_dir.exists():
    pred_files = list(predictions_dir.glob('predictions_*.csv'))
    
    if pred_files:
        log_print(f"\n--- Found {len(pred_files)} Prediction Files ---")
        
        for pred_file in pred_files:
            log_print(f"\nAnalyzing: {pred_file.name}")
            pred_df = pd.read_csv(pred_file)
            
            if 'Risk_Level' in pred_df.columns:
                risk_dist = pred_df['Risk_Level'].value_counts()
                log_print(f"  Risk Distribution:")
                for risk, count in risk_dist.items():
                    pct = count / len(pred_df) * 100
                    log_print(f"    {risk}: {count:,} ({pct:.1f}%)")
        
        # Visualize latest predictions (12M)
        pred_12m_path = predictions_dir / 'predictions_12m.csv'
        if pred_12m_path.exists():
            pred_df = pd.read_csv(pred_12m_path)
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Risk distribution
            if 'Risk_Level' in pred_df.columns:
                ax = axes[0]
                risk_dist = pred_df['Risk_Level'].value_counts()
                colors = {'Critical': 'red', 'High': 'orange', 'Medium': 'yellow', 'Low': 'green'}
                plot_colors = [colors.get(risk, 'gray') for risk in risk_dist.index]
                
                ax.bar(risk_dist.index, risk_dist.values, color=plot_colors, alpha=0.7)
                ax.set_xlabel('Risk Level', fontsize=11)
                ax.set_ylabel('Number of Equipment', fontsize=11)
                ax.set_title('Model Predictions - 12M Risk Distribution', fontsize=13, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
            
            # Risk score distribution
            if 'Risk_Score' in pred_df.columns:
                ax = axes[1]
                ax.hist(pred_df['Risk_Score'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
                ax.set_xlabel('Risk Score', fontsize=11)
                ax.set_ylabel('Number of Equipment', fontsize=11)
                ax.set_title('Model Predictions - 12M Risk Score Distribution', fontsize=13, fontweight='bold')
                ax.axvline(50, color='red', linestyle='--', label='High Risk Threshold')
                ax.legend()
                ax.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('outputs/eda/10_model_predictions.png', dpi=300, bbox_inches='tight')
            plt.close()
            log_print("\n✓ Saved: outputs/eda/10_model_predictions.png")
    else:
        log_print("\n⚠️  No prediction files found. Run 06_model_training.py first to see prediction analysis.")
else:
    log_print("\n⚠️  Predictions directory not found. Run 06_model_training.py first.")

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

log_print("\n" + "="*100)
log_print("EDA PIPELINE COMPLETE")
log_print("="*100)

# Close report file
report_file.close()
print(f"\n✅ Full report saved: reports/eda_summary.txt")