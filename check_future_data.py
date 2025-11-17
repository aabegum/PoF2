"""
CHECK IF YOU HAVE FUTURE FAILURE DATA FOR TEMPORAL TARGETS
Run this on your Windows machine to determine if temporal PoF is feasible
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CHECKING FUTURE FAILURE DATA AVAILABILITY")
print("="*80)

# Load original data
print("\nLoading data/combined_data.xlsx...")
df = pd.read_excel('data/combined_data.xlsx')
df['started at'] = pd.to_datetime(df['started at'], dayfirst=True, errors='coerce')

# Check date range
print(f'\nEarliest failure: {df["started at"].min()}')
print(f'Latest failure:   {df["started at"].max()}')
print(f'Total faults:     {len(df):,}')

# Check against cutoff
cutoff = pd.Timestamp('2024-06-25')
future_6m = cutoff + pd.DateOffset(months=6)   # 2024-12-25
future_12m = cutoff + pd.DateOffset(months=12)  # 2025-06-25

print(f'\n--- Temporal Target Windows ---')
print(f'Cutoff date:      {cutoff.date()}')
print(f'6M target end:    {future_6m.date()}')
print(f'12M target end:   {future_12m.date()}')

# Count future failures
faults_after_cutoff = df[df['started at'] > cutoff]
faults_in_6m = df[(df['started at'] > cutoff) & (df['started at'] <= future_6m)]
faults_in_12m = df[(df['started at'] > cutoff) & (df['started at'] <= future_12m)]

print(f'\nFaults after cutoff (2024-06-25): {len(faults_after_cutoff):,}')
print(f'Faults in 6M window:  {len(faults_in_6m):,}')
print(f'Faults in 12M window: {len(faults_in_12m):,}')

# Equipment counts
if len(faults_in_6m) > 0:
    equip_6m = faults_in_6m['cbs_id'].dropna().nunique()
else:
    equip_6m = 0

if len(faults_in_12m) > 0:
    equip_12m = faults_in_12m['cbs_id'].dropna().nunique()
else:
    equip_12m = 0

print(f'\nUnique equipment with failures:')
print(f'  In 6M window:  {equip_6m:,} equipment')
print(f'  In 12M window: {equip_12m:,} equipment')

# Determine feasibility
print(f'\n{"="*80}')
print("RECOMMENDATION")
print("="*80)

if len(faults_in_6m) >= 50:
    print('✅ 6M TEMPORAL TARGETS: FEASIBLE')
    print(f'   You have {equip_6m} equipment with future failures')
    print(f'   Expected positive class: ~{equip_6m/789*100:.1f}%')
else:
    print('❌ 6M TEMPORAL TARGETS: NOT FEASIBLE')
    print(f'   Only {len(faults_in_6m)} future faults - insufficient for training')

if len(faults_in_12m) >= 100:
    print('\n✅ 12M TEMPORAL TARGETS: FEASIBLE')
    print(f'   You have {equip_12m} equipment with future failures')
    print(f'   Expected positive class: ~{equip_12m/789*100:.1f}%')
else:
    print('\n❌ 12M TEMPORAL TARGETS: NOT FEASIBLE')
    print(f'   Only {len(faults_in_12m)} future faults - insufficient for training')

print(f'\n{"="*80}')
print("NEXT STEPS")
print("="*80)

if len(faults_after_cutoff) >= 100:
    print('\n✅ SUFFICIENT FUTURE DATA - Implement temporal PoF approach!')
    print('\nActions:')
    print('  1. Update script 06 with temporal target creation code')
    print('  2. Re-run pipeline: python 06_model_training.py')
    print('  3. Expect AUC ~0.75-0.85 (realistic, not 1.0)')
    print('  4. Validate predictions against actual outcomes')
else:
    print('\n⚠️  INSUFFICIENT FUTURE DATA - Use Survival Analysis instead!')
    print('\nActions:')
    print('  1. Keep current chronic repeater model (script 06)')
    print('  2. Run survival analysis: python 09_survival_analysis.py')
    print('  3. Cox model provides temporal PoF WITHOUT needing future labels')
    print('  4. Combines both approaches for complete PoF prediction')

print(f'\n{"="*80}')
