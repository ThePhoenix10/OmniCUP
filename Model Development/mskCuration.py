import pandas as pd
import numpy as np
import re

print("Step 1: Loading data...")

paths = {
    'sample': '/content/drive/MyDrive/Research/Datasets/msk_met_2021/data_clinical_sample.txt',
    'patient': '/content/drive/MyDrive/Research/Datasets/msk_met_2021/data_clinical_patient.txt',
    'mut': '/content/drive/MyDrive/Research/Datasets/msk_met_2021/data_mutations.txt',
    'cna': '/content/drive/MyDrive/Research/Datasets/msk_met_2021/data_cna.txt',
    'sv': '/content/drive/MyDrive/Research/Datasets/msk_met_2021/data_sv.txt',
    'hotspots': '/content/drive/MyDrive/Research/Datasets/hotspots_v2.xls'
}

df_sample  = pd.read_csv(paths['sample'], sep='\t', comment='#')
df_patient = pd.read_csv(paths['patient'], sep='\t', comment='#')
df_mut     = pd.read_csv(paths['mut'], sep='\t', comment='#', low_memory=False)
df_cna     = pd.read_csv(paths['cna'], sep='\t', comment='#')
df_sv      = pd.read_csv(paths['sv'], sep='\t', comment='#')
hotspot_df = pd.read_excel(paths['hotspots'])

print("Step 2: Merging clinical metadata and processing distant metastases...")

df_main = pd.merge(df_sample, df_patient, on='PATIENT_ID', how='left')

clinical_cols = [
    'SAMPLE_ID',
    'CANCER_TYPE',
    'SEX',
    'AGE_AT_SEQUENCING',
    'TMB_NONSYNONYMOUS',
    'MSI_SCORE'
]

dmets_cols = [c for c in df_main.columns if c.startswith("DMETS_DX_")]
clinical_cols.extend(dmets_cols)

for c in clinical_cols:
    if c not in df_main.columns:
        df_main[c] = np.nan

df_main = df_main[clinical_cols].copy()

df_main['SEX'] = df_main['SEX'].map({'Male': 1, 'Female': 0, 'M': 1, 'F': 0})
df_main = df_main.rename(columns={'CANCER_TYPE': 'Primary_Site_Target'})

for col in dmets_cols:
    df_main[col] = (
        df_main[col]
        .map({"Yes": 1, "No": 0, "YES": 1, "NO": 0})
        .fillna(0)
        .astype(int)
    )

for col in ['AGE_AT_SEQUENCING', 'TMB_NONSYNONYMOUS', 'MSI_SCORE']:
    if col in df_main.columns:
        df_main[col] = df_main[col].fillna(df_main[col].median())

print("Step 3: Classifying somatic mutation types...")

som_mat_subcategories = [
    'Missense_Mutation',
    'Nonsense_Mutation',
    'Frame_Shift_Ins',
    'Frame_Shift_Del',
    'Splice_Site',
    'Translation_Start_Site',
    'In_Frame_Ins',
    'In_Frame_Del',
    'Nonstop_Mutation'
]

print("\n--- All Somatic Mutation Subcategories in data_mutations.txt ---")

variant_counts = (
    df_mut["Variant_Classification"]
    .fillna("MISSING")
    .astype(str)
    .value_counts()
)

print(f"{'Subcategory':<35} | {'Count':<10} | {'Used in ANY_SOM_MAT'}")
print("-" * 75)

for subcategory, count in variant_counts.items():
    used_flag = "YES" if subcategory in som_mat_subcategories else "NO"
    print(f"{subcategory:<35} | {count:<10} | {used_flag}")

print("-" * 75)
print(f"Total unique somatic mutation subcategories found: {len(variant_counts)}")
print(f"Somatic mutation subcategories included in model: {len(som_mat_subcategories)}")

df_mut['is_ANY_SOM_MAT'] = (
    df_mut['Variant_Classification']
    .isin(som_mat_subcategories)
    .astype(int)
)

for som_mat_class in som_mat_subcategories:
    df_mut[f'is_{som_mat_class}'] = (
        df_mut['Variant_Classification'] == som_mat_class
    ).astype(int)

print("Step 4: Processing curated hotspots...")

hotspot_df = hotspot_df[
    [
        'Hugo_Symbol',
        'Amino_Acid_Position',
        'Reference_Amino_Acid',
        'Variant_Amino_Acid'
    ]
].dropna()

hotspot_df['Amino_Acid_Position'] = hotspot_df['Amino_Acid_Position'].astype(str)

aa_ref = hotspot_df[hotspot_df['Amino_Acid_Position'].str.match(r'^\d+$')].copy()
aa_ref['ref'] = aa_ref['Reference_Amino_Acid'].str.split(':').str[0]
aa_ref['alt'] = aa_ref['Variant_Amino_Acid'].str.split(':').str[0]
aa_ref['KEY'] = (
    aa_ref['Hugo_Symbol'] + '.' +
    aa_ref['ref'] +
    aa_ref['Amino_Acid_Position'] +
    aa_ref['alt']
)

curated_aa_set = set(aa_ref['KEY'])

df_mut['MUT_AA_KEY'] = (
    df_mut['Hugo_Symbol'] + '.' +
    df_mut['HGVSp_Short'].str.replace('p.', '', regex=False)
)

df_mut['is_curated_aa'] = df_mut['MUT_AA_KEY'].isin(curated_aa_set).astype(int)

splice_ref = hotspot_df[
    hotspot_df['Amino_Acid_Position'].str.contains('splice', case=False)
].copy()

splice_ref['Codon'] = splice_ref['Amino_Acid_Position'].str.extract(r'(\d+)')
splice_ref['KEY'] = splice_ref['Hugo_Symbol'] + '.X' + splice_ref['Codon'] + '_splice'

curated_splice_set = set(splice_ref['KEY'])

df_mut['MUT_SPLICE_KEY'] = (
    df_mut['Hugo_Symbol'] + '.' +
    df_mut['HGVSp_Short']
    .str.extract(r'(X\d+_splice)', expand=False)
    .fillna('')
)

df_mut['is_curated_splice'] = df_mut['MUT_SPLICE_KEY'].isin(curated_splice_set).astype(int)

df_mut['is_any_hotspot'] = (
    (df_mut['is_curated_aa'] == 1) |
    (df_mut['is_curated_splice'] == 1)
).astype(int)

print("Step 5: Building feature matrices...")

any_som_mat_matrix = df_mut[df_mut.is_ANY_SOM_MAT == 1].pivot_table(
    index='Tumor_Sample_Barcode',
    columns='Hugo_Symbol',
    values='is_ANY_SOM_MAT',
    aggfunc='max'
).fillna(0).astype(int).add_suffix('_ANY_SOM_MAT')

som_mat_subcategory_matrices = []

for som_mat_class in som_mat_subcategories:
    flag_col = f'is_{som_mat_class}'
    df_sub = df_mut[df_mut[flag_col] == 1]

    if not df_sub.empty:
        mat = df_sub.pivot_table(
            index='Tumor_Sample_Barcode',
            columns='Hugo_Symbol',
            values=flag_col,
            aggfunc='max'
        ).fillna(0).astype(int).add_suffix(f'_{som_mat_class}')

        som_mat_subcategory_matrices.append(mat)

gene_hotspot_matrix = df_mut[df_mut.is_any_hotspot == 1].pivot_table(
    index='Tumor_Sample_Barcode',
    columns='Hugo_Symbol',
    values='is_any_hotspot',
    aggfunc='max'
).fillna(0).astype(int).add_suffix('_HOTSPOT')

aa_hotspot_matrix = df_mut[df_mut.is_curated_aa == 1].pivot_table(
    index='Tumor_Sample_Barcode',
    columns='MUT_AA_KEY',
    values='is_curated_aa',
    aggfunc='max'
).fillna(0).astype(int).add_suffix('_AA_HOTSPOT')

splice_hotspot_matrix = df_mut[df_mut.is_curated_splice == 1].pivot_table(
    index='Tumor_Sample_Barcode',
    columns='MUT_SPLICE_KEY',
    values='is_curated_splice',
    aggfunc='max'
).fillna(0).astype(int).add_suffix('_SPLICE_HOTSPOT')

df_cna_p = df_cna.set_index('Hugo_Symbol').T

amp_matrix = (df_cna_p == 2).astype(int).add_suffix('_AMP')
del_matrix = (df_cna_p == -2).astype(int).add_suffix('_DEL')
any_cna_matrix = ((df_cna_p == 2) | (df_cna_p == -2)).astype(int).add_suffix('_ANY_CNA')

df_fusion = pd.concat([
    df_sv[['Sample_Id', 'Site1_Hugo_Symbol']]
    .rename(columns={'Site1_Hugo_Symbol': 'Hugo_Symbol'}),

    df_sv[['Sample_Id', 'Site2_Hugo_Symbol']]
    .rename(columns={'Site2_Hugo_Symbol': 'Hugo_Symbol'})
]).dropna()

df_fusion['is_fusion'] = 1

sv_matrix = df_fusion.pivot_table(
    index='Sample_Id',
    columns='Hugo_Symbol',
    values='is_fusion',
    aggfunc='max'
).fillna(0).astype(int).add_suffix('_FUSION')

print("Step 6: Executing final join...")

matrices_to_join = [any_som_mat_matrix] + som_mat_subcategory_matrices + [
    amp_matrix,
    del_matrix,
    any_cna_matrix,
    sv_matrix,
    gene_hotspot_matrix,
    aa_hotspot_matrix,
    splice_hotspot_matrix
]

final_df = df_main.set_index('SAMPLE_ID').join(matrices_to_join, how='left').fillna(0)

clinical_order = [
    'Primary_Site_Target',
    'AGE_AT_SEQUENCING',
    'SEX',
    'TMB_NONSYNONYMOUS',
    'MSI_SCORE'
] + dmets_cols

clinical_order = [c for c in clinical_order if c in final_df.columns]

genomic_cols = [c for c in final_df.columns if c not in clinical_order]

final_df = final_df[clinical_order + genomic_cols]

final_df.insert(
    len(clinical_order),
    'IS_ANY_SOM_MAT',
    (final_df.filter(regex='_ANY_SOM_MAT$').sum(axis=1) > 0).astype(int)
)

final_df.to_csv('MSKMET_Curated_Final.csv')

print("\n--- Feature Type Examples Verification ---")

def get_example(pattern):
    cols = [c for c in final_df.columns if re.search(pattern, c)]

    for c in cols:
        if (final_df[c] == 1).any():
            return c

    return "None found"

print(f"{'Category':<35} | {'Example Feature Name'}")
print("-" * 80)

print(f"{'Distant Metastasis DX':<35} | {get_example(r'^DMETS_DX_')}")
print(f"{'Main ANY_SOM_MAT':<35} | {get_example(r'_ANY_SOM_MAT$')}")
print(f"{'Missense Mutation':<35} | {get_example(r'_Missense_Mutation$')}")
print(f"{'Nonsense Mutation':<35} | {get_example(r'_Nonsense_Mutation$')}")
print(f"{'Frame Shift Insertion':<35} | {get_example(r'_Frame_Shift_Ins$')}")
print(f"{'Frame Shift Deletion':<35} | {get_example(r'_Frame_Shift_Del$')}")
print(f"{'Splice Site':<35} | {get_example(r'_Splice_Site$')}")
print(f"{'Translation Start Site':<35} | {get_example(r'_Translation_Start_Site$')}")
print(f"{'In Frame Insertion':<35} | {get_example(r'_In_Frame_Ins$')}")
print(f"{'In Frame Deletion':<35} | {get_example(r'_In_Frame_Del$')}")
print(f"{'Nonstop Mutation':<35} | {get_example(r'_Nonstop_Mutation$')}")
print(f"{'Amplification':<35} | {get_example(r'_AMP$')}")
print(f"{'Deletion':<35} | {get_example(r'_DEL$')}")
print(f"{'Any CNA AMP or DEL':<35} | {get_example(r'_ANY_CNA$')}")
print(f"{'Fusion':<35} | {get_example(r'_FUSION$')}")
print(f"{'Gene-level Hotspot':<35} | {get_example(r'^[^.]+(_HOTSPOT)$')}")
print(f"{'Allele-level AA Hotspot':<35} | {get_example(r'_AA_HOTSPOT$')}")
print(f"{'Splice Hotspot':<35} | {get_example(r'_SPLICE_HOTSPOT$')}")

feature_counts = {
    "DISTANT_METASTASES": len(dmets_cols),

    "ANY_SOM_MAT": len([
        c for c in final_df.columns
        if c.endswith("_ANY_SOM_MAT")
    ]),

    "SOMATIC_MUTATION_SUBCATEGORIES": len([
        c for c in final_df.columns
        if any(c.endswith(f"_{mc}") for mc in som_mat_subcategories)
    ]),

    "MISSENSE_MUTATION": len([
        c for c in final_df.columns
        if c.endswith("_Missense_Mutation")
    ]),

    "NONSENSE_MUTATION": len([
        c for c in final_df.columns
        if c.endswith("_Nonsense_Mutation")
    ]),

    "FRAME_SHIFT_INS": len([
        c for c in final_df.columns
        if c.endswith("_Frame_Shift_Ins")
    ]),

    "FRAME_SHIFT_DEL": len([
        c for c in final_df.columns
        if c.endswith("_Frame_Shift_Del")
    ]),

    "SPLICE_SITE": len([
        c for c in final_df.columns
        if c.endswith("_Splice_Site")
    ]),

    "TRANSLATION_START_SITE": len([
        c for c in final_df.columns
        if c.endswith("_Translation_Start_Site")
    ]),

    "IN_FRAME_INS": len([
        c for c in final_df.columns
        if c.endswith("_In_Frame_Ins")
    ]),

    "IN_FRAME_DEL": len([
        c for c in final_df.columns
        if c.endswith("_In_Frame_Del")
    ]),

    "NONSTOP_MUTATION": len([
        c for c in final_df.columns
        if c.endswith("_Nonstop_Mutation")
    ]),

    "AMP": len([
        c for c in final_df.columns
        if c.endswith("_AMP")
    ]),

    "DEL": len([
        c for c in final_df.columns
        if c.endswith("_DEL")
    ]),

    "ANY_CNA": len([
        c for c in final_df.columns
        if c.endswith("_ANY_CNA")
    ]),

    "FUSION": len([
        c for c in final_df.columns
        if c.endswith("_FUSION")
    ]),

    "GENE_HOTSPOT": len([
        c for c in final_df.columns
        if c.endswith("_HOTSPOT")
        and "_AA_" not in c
        and "_SPLICE_" not in c
    ]),

    "AA_HOTSPOT": len([
        c for c in final_df.columns
        if c.endswith("_AA_HOTSPOT")
    ]),

    "SPLICE_HOTSPOT": len([
        c for c in final_df.columns
        if c.endswith("_SPLICE_HOTSPOT")
    ])
}

print("\n--- Genomic Feature Counts ---")

for k, v in feature_counts.items():
    print(f"{k:<35}: {v}")