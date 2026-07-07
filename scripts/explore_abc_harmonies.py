import pandas as pd

harm = pd.read_csv(r"external\ABC\harmonies\n11op95_01.harmonies.tsv", sep='\t')
print("列名:", harm.columns.tolist())
print("\n前 20 行:")
print(harm.head(20))
print(f"\n总行数: {len(harm)}")

if 'cadence' in harm.columns:
    cad = harm[harm['cadence'].notna()]
    print(f"\ncadence 数量: {len(cad)}")
    cols_to_show = [c for c in ['mn', 'mc_onset', 'cadence', 'chord', 'globalkey', 'localkey'] if c in harm.columns]
    print(cad[cols_to_show].head(20))
    
if 'phraseend' in harm.columns:
    ph = harm[harm['phraseend'].notna()]
    print(f"\nphraseend 数量: {len(ph)}")
    print(ph[['mn', 'quarterbeats', 'phraseend', 'chord']].head(20))

# 找 localkey 变化点
harm['localkey_prev'] = harm['localkey'].shift(1)
key_changes = harm[harm['localkey'] != harm['localkey_prev']].copy()
print(f"\nlocalkey 变化次数: {len(key_changes)}")
print(key_changes[['mn', 'quarterbeats', 'localkey', 'globalkey']].head(20))