#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import scanpy as sc
import seaborn as sns
import numpy as np
import h5py
import pandas as pd
import scanpy as sc
import scanpy.external as sce
import matplotlib as plt

# Core scientific libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
from scipy.stats import median_abs_deviation



# Plotting settings
warnings.simplefilter(action='ignore', category=FutureWarning)
sc.settings.verbosity = 0
sns.set(rc={"figure.figsize": (4, 3.5), "figure.dpi": 100})
sns.set_style("whitegrid")

sc.settings.set_figure_params(
    dpi=300,        # inline resolution
    dpi_save=300,   # saved figures
    frameon=False
)


# In[2]:


get_ipython().system('pwd')


# In[2]:


import os

# Confirm it's set
#print("Current working directory:", os.getcwd())

import os
import os
os.environ["OPENBLAS_NUM_THREADS"] = "25"
os.environ["OMP_NUM_THREADS"] = "25"
os.environ["MKL_NUM_THREADS"] = "25"

# optional: lock NumExpr too
os.environ["NUMEXPR_MAX_THREADS"] = "25"
os.environ["NUMEXPR_NUM_THREADS"] = "25"
os.environ["R_HOME"] = "/home/grathore/conda_envs/gaurav_python_r/lib/R"

# Load R magic
# %load_ext rpy2.ipython

# Core scientific libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
from scipy.stats import median_abs_deviation


os.environ["OPENBLAS_NUM_THREADS"] = "25"
os.environ["OMP_NUM_THREADS"] = "25"
os.environ["MKL_NUM_THREADS"] = "25"

# Single-cell libs
import scanpy as sc
import loompy as lp

# import anndata2ri
# import scanpy.external as sce

# # rpy2 conversion setup
# import rpy2.robjects as ro
# from rpy2.robjects.conversion import localconverter
# from rpy2.robjects import pandas2ri
# import rpy2.robjects.conversion as conversion
# import rpy2.rinterface_lib.callbacks as rcb

# # Enable automatic conversion via setting conversion functions explicitly
# conversion.py2rpy = pandas2ri.py2rpy
# conversion.rpy2py = pandas2ri.rpy2py

# # Enable AnnData <-> R conversion
# anndata2ri.activate()

# rcb.logger.setLevel(logging.ERROR)
# #ro.pandas2ri.activate()
# anndata2ri.activate()

# Plotting settings
warnings.simplefilter(action='ignore', category=FutureWarning)
sc.settings.verbosity = 0
sns.set(rc={"figure.figsize": (4, 3.5), "figure.dpi": 100})
sns.set_style("whitegrid")

# sc.settings.set_figure_params(
#     dpi=300,        # inline resolution
#     dpi_save=300,   # saved figures
#     frameon=False
# )


# In[4]:


# adata = sc.read('/mnt/nfs/CX000008_DS1/projects/gaurav/published_datasets/developing_human/linnarsson/ft_transcriptome/hbd_atlas.h5ad')
# adata.var_names_make_unique()


# In[5]:


# # make sure age is string (important)
# adata.obs["Age"] = adata.obs["Age"].astype(str)

# # define ages to highlight
# keep_ages = ["5.0", "5.5"]

# # create highlight column
# adata.obs["age_highlight"] = np.where(
#     adata.obs["Age"].isin(keep_ages),
#     adata.obs["Age"],
#     "other"
# )

# # sc.pl.embedding(
# #     adata,
# #     basis = "X_embedding",
# #     color="age_highlight",
# #     palette={
# #         "5.0": "#1f77b4",
# #         "5.5": "#ff7f0e",
# #         "other": "lightgrey",
# #     },
# #     frameon=False,
# #     title="Age 5.0 & 5.5 highlighted",
# # )


# In[6]:


# adata_5 = adata[adata.obs["Age"].astype(str).isin(["5.0", "5.5"])].copy()
# adata_5.write('/mnt/nfs/CX000008_DS1/projects/gaurav/published_datasets/developing_human/linnarsson/ft_transcriptome/hbd_atlas_age_pcw5_subset.h5ad')


# In[7]:


# mistr = sc.read_h5ad('/mnt/nfs/CX000008_DS1/projects/gaurav/mistr/published_dataset_analyis/mistr_all_leiden_annotated_final_gsr.h5ad')

# var_names = mistr.var_names
# len(var_names)
# with open("mistr_var_names.txt", "w") as f:
#     for g in var_names:
#         f.write(f"{g}\n")


# ## read week5 pc human data braun et al

# In[3]:


adata = sc.read('/mnt/nfs/CX000008_DS1/projects/gaurav/published_datasets/developing_human/linnarsson/ft_transcriptome/hbd_atlas_age_pcw5_subset.h5ad')
sc.pp.subsample(adata, fraction=0.5)
adata


# ## ensure braun data has same genes as MISTR data

# In[4]:


# load
with open("mistr_var_names.txt") as f:
    var_names_mistr = [line.strip() for line in f]

# convert to Index once
var_names_mistr = pd.Index(var_names_mistr)

# intersect (order follows adata.var_names)
common_genes = adata.var_names.intersection(var_names_mistr)

# subset adata
adata = adata[:, common_genes].copy()

len(common_genes)

adata


# In[5]:


# 1) Sanity: make sure your clustering column exists
assert "Tissue" in adata.obs.columns

# 2) Ensure RAW COUNTS are available
# Prefer: a counts layer
if "counts" in adata.layers:
    # store counts into .raw (SCENIC expects raw-like access)
    adata.raw = adata.copy()
    adata.raw.X = adata.layers["counts"].copy()
else:
    # if you don't have a counts layer, we assume adata.X is counts
    # (ONLY ok if you haven't log-normalized yet)
    adata.raw = adata.copy()

# 3) Quick checks
print("X min/max:", float(adata.X.min()), float(adata.X.max()))
print("raw.X min/max:", float(adata.raw.X.min()), float(adata.raw.X.max()))
print("n_cells, n_genes:", adata.n_obs, adata.n_vars)
print("Tissue n:", adata.obs["Tissue"].nunique())
print(adata.obs["Tissue"].value_counts().head(10))


# In[6]:


import scanpy as sc

# --- Step 2: Filtering for SCENIC (counts) ---
adata_scn = adata.copy()

# put raw counts into X (SCENIC wants counts)
adata_scn.X = adata.raw.X.copy()

# basic QC filters (good defaults)
sc.pp.filter_genes(adata_scn, min_cells=20)
sc.pp.filter_cells(adata_scn, min_genes=200)

print("After filtering:", adata_scn.n_obs, adata_scn.n_vars)

# keep only the clustering label you'll use
# adata_scn.obs = adata_scn.obs[["Tissue"]].copy()
# print("Tissue counts:")
# print(adata_scn.obs["Tissue"].value_counts())


# In[7]:


# set variables for file paths to read from and write to:

# set a working directory
wdir = "/mnt/nfs/CX000008_DS1/projects/gaurav/mistr/published_dataset_analyis/linnarsson_hbd_scenic/"
os.chdir( wdir )

# # path to loom file with basic filtering applied (this will be created in the "initial filtering" step below). Optional.
f_loom_path_scenic = "hbd_pc5.loom"

f_matrix_path_scenic = 'exp_matrix.csv'

# path to anndata object, which will be updated to store Scanpy results as they are generated below
f_anndata_path = "hbd_pc5.h5ad"

# path to pyscenic output
f_pyscenic_output = "hbd_pc5_pyscenic_output.loom"

# loom output, generated from a combination of Scanpy and pySCENIC results:
f_final_loom = 'hbd_pc5_scenic_integrated-output.loom'


# In[13]:


# ### CELL 4 — Normalize, log1p, HVGs (per sample), PCA

# import scanpy as sc
# import numpy as np

#  # just to be explicit

# print("=== RNA modality BEFORE normalization ===")
# print(f"  Cells (n_obs): {adata_scn.n_obs}")
# print(f"  Genes (n_vars): {adata_scn.n_vars}")
# print(f"  .obs keys: {list(adata_scn.obs.columns)}\n")

# # 1) Make sure raw counts are preserved in layers['counts']
# # if "counts" not in adata.layers:
# #     print("rna.layers['counts'] not found — creating it from current X.")
# #     rna.layers["counts"] = rna.X.copy()
# # else:
# #     print("Found rna.layers['counts'] (raw counts).")
# adata_scn.layers["counts"] = adata_scn.X.copy()
# # 2) Normalize and log-transform X (this is what we'll use for PCA/UMAP)
# print("\nNormalizing total counts per cell to 1e4 and log1p-transforming...")
# sc.pp.normalize_total(adata_scn, target_sum=1e4)
# sc.pp.log1p(adata_scn)
# # 3) Highly variable genes (Seurat v3, per sample)
# # if "sample" not in adata.obs.columns:
# #     raise ValueError("adata.obs['sample'] is missing; needed as batch_key for HVGs.")

# # print("\nSelecting highly variable genes with Seurat v3 (batch_key='sample')...")
# sc.pp.highly_variable_genes(
#     adata_scn,
#     flavor="seurat_v3",
#     layer ="counts",
#     n_top_genes=3000,
#     #batch_key="sample"
#     subset= True
# )

# n_hvg = int(adata_scn.var["highly_variable"].sum())
# print(f"  Total HVGs selected: {n_hvg}")

# # sanity: how many genes per sample are HVG (rough idea)
# print("\nExample of HVG fraction (first 5 genes):")
# print(adata_scn.var[["highly_variable"]].head())
# print()

# # 4) Subset to HVGs for PCA / neighbors
# rna_hvg = adata_scn[:, adata_scn.var["highly_variable"]].copy()
# print(f"rna_hvg shape (cells x HVGs): {rna_hvg.n_obs} x {rna_hvg.n_vars}\n")

# # 5) PCA
# print("Running PCA on HVGs (50 components)...")
# sc.tl.pca(rna_hvg, n_comps=50, svd_solver="arpack")

# expl = rna_hvg.uns["pca"]["variance_ratio"]
# print("  Explained variance by first 10 PCs:")
# for i, v in enumerate(expl[:10]):
#     print(f"    PC{i+1}: {v*100:.2f}%")
# print(f"  Cumulative (PC1–10): {expl[:10].sum()*100:.2f}%\n")

# print("PCA finished. rna_hvg.obsm['X_pca'] is ready.")



# In[8]:


### CELL 4 — Normalize, log1p, HVGs (3k) + force markers, then subset, then PCA

import scanpy as sc
import numpy as np

FORCE_MARKERS = ["OTX2", "GBX2", "MSX1", "PAX8", "PAX5", "PAX2", "EN1"]

print("=== RNA modality BEFORE normalization ===")
print(f"  Cells (n_obs): {adata_scn.n_obs}")
print(f"  Genes (n_vars): {adata_scn.n_vars}")
print(f"  .obs keys: {list(adata_scn.obs.columns)}\n")

# 1) Save raw counts
adata_scn.layers["counts"] = adata_scn.X.copy()

# 2) Normalize + log1p for PCA/UMAP inputs
print("Normalizing total counts per cell to 1e4 and log1p-transforming...")
sc.pp.normalize_total(adata_scn, target_sum=1e4)
sc.pp.log1p(adata_scn)

# 3) Compute HVGs (do NOT subset yet)
print("Selecting 3000 HVGs (Seurat v3)...")
sc.pp.highly_variable_genes(
    adata_scn,
    flavor="seurat_v3",
    layer="counts",
    n_top_genes=3000,
    subset=False,
    # batch_key="sample",  # uncomment if you really want per-sample HVGs
)

# 4) Build final keep-list = 3000 HVGs + forced markers (only if present)
hvg_genes = adata_scn.var_names[adata_scn.var["highly_variable"]].tolist()
hvg_set = set(hvg_genes)

present_markers = [g for g in FORCE_MARKERS if g in adata_scn.var_names]
missing_markers = [g for g in FORCE_MARKERS if g not in adata_scn.var_names]

forced_to_add = [g for g in present_markers if g not in hvg_set]
keep_genes = hvg_genes + forced_to_add  # 3000 + extras only

print(f"  HVGs: {len(hvg_genes)} (should be 3000)")
print(f"  Forced markers present: {present_markers}")
print(f"  Forced markers added (not already HVG): {forced_to_add}")
print(f"  Forced markers missing (not in var_names): {missing_markers}")
print(f"  Final gene count kept: {len(keep_genes)}\n")

# 5) Subset immediately to keep dataset small
rna_hvg = adata_scn[:, keep_genes].copy()
adata_scn = adata_scn[:, keep_genes].copy()
print(f"rna_hvg shape (cells x genes): {rna_hvg.n_obs} x {rna_hvg.n_vars}\n")

# 6) PCA
print("Running PCA (50 components)...")
sc.tl.pca(rna_hvg, n_comps=50, svd_solver="arpack")

expl = rna_hvg.uns["pca"]["variance_ratio"]
print("  Explained variance by first 10 PCs:")
for i, v in enumerate(expl[:10]):
    print(f"    PC{i+1}: {v*100:.2f}%")
print(f"  Cumulative (PC1–10): {expl[:10].sum()*100:.2f}%\n")

print("PCA finished. rna_hvg.obsm['X_pca'] is ready.")


# In[10]:


### CELL 5 — Neighbors + UMAP on PCA space

print("=== Building neighbors graph on PCA space ===")
sc.pp.neighbors(
    rna_hvg,
    n_neighbors=50,   # adjust if you want more global vs local structure
    n_pcs=45
    #metric="cosine"
)

print("Neighbors graph computed:")
print("  rna_hvg.obsp['connectivities'].shape =", rna_hvg.obsp["connectivities"].shape)
print("  rna_hvg.obsp['distances'].shape      =", rna_hvg.obsp["distances"].shape, "\n")

print("Running UMAP (min_dist=0.4, spread=1.0)...")
sc.tl.umap(
    rna_hvg,
    min_dist=0.5,     # larger = smoother, less clumpy; smaller = more fine detail
    spread=0.5,
    random_state=0
)

print("UMAP finished. rna_hvg.obsm['X_umap'] is ready.\n")

#Copy PCA, neighbors, UMAP back to full RNA object for convenience
adata_scn.obsm["X_pca"] = rna_hvg.obsm["X_pca"]
adata_scn.obsp["connectivities"] = rna_hvg.obsp["connectivities"]
adata_scn.obsp["distances"]      = rna_hvg.obsp["distances"]
adata_scn.obsm["X_umap"]         = rna_hvg.obsm["X_umap"]

print("=== Finished: RNA modality now has PCA, neighbors, and UMAP stored. ===")
print(f"  adata_scn.obsm['X_umap'].shape = {adata_scn.obsm['X_umap'].shape}")


# In[11]:


# create basic row and column attributes for the loom file:

rna_hvg.x = rna_hvg.layers['counts'].copy()
row_attrs = {
    "Gene": np.array(rna_hvg.var_names) ,
}
col_attrs = {
    "CellID": np.array(rna_hvg.obs_names) ,
    "nGene": np.array( np.sum(rna_hvg.X.transpose()>0 , axis=0)).flatten() ,
    "nUMI": np.array( np.sum(rna_hvg.X.transpose() , axis=0)).flatten() ,
}
lp.create( f_loom_path_scenic, rna_hvg.X.transpose(), row_attrs, col_attrs)


# In[12]:


sc.pl.umap(adata_scn, color=["Subregion","Age"], ncols=2)


# In[13]:


sc.pl.umap(adata_scn, color = ['OTX2',"GBX2","FGF17","FGF8","EN1","WNT1","FEZF1","FOXG1","STMN2"], cmap='cividis', ncols=5, vmax= "p99" )


# In[15]:


import os, glob, re, pickle
import scanpy as sc
import pandas as pd
import numpy as np
import scanpy.external as sce
import re
from sklearn.metrics import pairwise_distances
import seaborn as sns
import matplotlib.pyplot as pl
import matplotlib.colors as colors
import operator as op
import anndata as ad
import loompy as lp
#import celloracle as co
import pyarrow
import matplotlib as plt



from functools import partial
from collections import OrderedDict
from cytoolz import compose
from pyscenic.export import export2loom, add_scenic_metadata
from pyscenic.utils import load_motifs
from pyscenic.transform import df2regulons
from pyscenic.aucell import aucell
from pyscenic.binarization import binarize
from pyscenic.rss import regulon_specificity_scores
from pyscenic.plotting import plot_binarization, plot_rss

from IPython.display import HTML, display



get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', None)
sc.set_figure_params(dpi = 150, dpi_save = 150, format = 'png')
sc._settings.ScanpyConfig(verbosity=0)


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

np.random.seed(44)


# # scenic step 1 GRN inference using Grnboost2

# In[16]:


## start scenic analysis 
f_tfs = "/mnt/nfs/CX000008_DS1/projects/gaurav/published_datasets/developing_human/scenic1/database/hg-tflist/hs_hgnc_tfs.txt" # human


# In[17]:


adata_scn.to_df().to_csv('exp_matrix.csv')


# In[18]:


get_ipython().system('pyscenic grn {f_loom_path_scenic} {f_tfs} -m grnboost2 -o adj.csv   --num_workers 50')


# In[19]:


######### read the adj file and look for TF targets
######### select target above mean expression of importance 
adjacencies = pd.read_csv("adj.csv", index_col=False)
value = np.quantile(adjacencies['importance'].copy(), q = 0.99)
adj2 = adjacencies.copy()
adj2 = adjacencies[adjacencies['importance']>= value]
adj2


# ## Regulon prediction aka cisTarget from CLI

# In[20]:


import glob
# ranking databases
f_db_glob = "/mnt/nfs/CX000008_DS1/projects/gaurav/published_datasets/developing_human/scenic1/database/cisTarget_databases/hg38/*feather"
f_db_names = ' '.join( glob.glob(f_db_glob) )

# motif databases
f_motif_path = "/mnt/nfs/CX000008_DS1/projects/gaurav/published_datasets/developing_human/scenic1/database/hg-anno/motifs-v9-nr.hgnc-m0.001-o0.0.tsv"


# In[21]:


get_ipython().system('pyscenic ctx adj.csv       {f_db_names}       --annotations_fname {f_motif_path}       --expression_mtx_fname {f_loom_path_scenic}       --output reg.csv       --mask_dropouts       --num_workers 10')


# In[22]:


nGenesDetectedPerCell = np.sum(adata_scn.X>0, axis=1)
percentiles = np.percentile(nGenesDetectedPerCell,[10,50, 75, 100])
print(percentiles)


# In[23]:


import numpy as np
import pandas as pd
import scipy.sparse as sp

X = adata_scn.X

if sp.issparse(X):
    nGenesDetectedPerCell = np.array((X > 0).sum(axis=1)).ravel()
else:
    nGenesDetectedPerCell = (X > 0).sum(axis=1)

nGenesDetectedPerCell = pd.Series(nGenesDetectedPerCell)

percentiles = nGenesDetectedPerCell.quantile([0.01, 0.05, 0.10, 0.50, 1.0])
percentiles


# In[24]:


fig, ax = plt.pyplot.subplots(1, 1, figsize=(8, 5), dpi=150)
sns.distplot(nGenesDetectedPerCell, norm_hist=False, kde=False, bins='fd')
for i,x in enumerate(percentiles):
    fig.gca().axvline(x=x, ymin=0,ymax=1, color='red')
    ax.text(x=x, y=ax.get_ylim()[1], s=f'{int(x)} ({percentiles.index.values[i]*100}%)', color='red', rotation=30, size='x-small',rotation_mode='anchor' )
ax.set_xlabel('# of genes')
ax.set_ylabel('# of cells')
fig.tight_layout()


# In[25]:


df_motifs = load_motifs('reg.csv')


# In[26]:


BASE_URL = "http://motifcollections.aertslab.org/v9/logos/"
COLUMN_NAME_LOGO = "MotifLogo"
COLUMN_NAME_MOTIF_ID = "MotifID"
COLUMN_NAME_TARGETS = "TargetGenes"


def display_logos(df: pd.DataFrame, top_target_genes: int = 3, base_url: str = BASE_URL):
    """
    :param df:
    :param base_url:
    """
    # Make sure the original dataframe is not altered.
    df = df.copy()
    
    # Add column with URLs to sequence logo.
    def create_url(motif_id):
        return '<img src="{}{}.png" style="max-height:124px;"></img>'.format(base_url, motif_id)
    df[("Enrichment", COLUMN_NAME_LOGO)] = list(map(create_url, df.index.get_level_values(COLUMN_NAME_MOTIF_ID)))
    
    # Truncate TargetGenes.
    def truncate(col_val):
        return sorted(col_val, key=op.itemgetter(1))[:top_target_genes]
    df[("Enrichment", COLUMN_NAME_TARGETS)] = list(map(truncate, df[("Enrichment", COLUMN_NAME_TARGETS)]))
    
    MAX_COL_WIDTH = pd.get_option('display.max_colwidth')
    pd.set_option('display.max_colwidth', -1)
    display(HTML(df.head().to_html(escape=False)))
    pd.set_option('display.max_colwidth', MAX_COL_WIDTH)
    
display_logos(df_motifs.head())


# In[ ]:





# In[27]:


def derive_regulons(motifs, db_names=('	hg38__refseq-r80__10kb_up_and_down_tss.mc9nr', 
                                 'hg38__refseq-r80__500bp_up_and_100bp_down_tss.mc9nr')):
    motifs.columns = motifs.columns.droplevel(0)

    def contains(*elems):
        def f(context):
            return any(elem in context for elem in elems)
        return f

    # For the creation of regulons we only keep the 10-species databases and the activating modules. We also remove the
    # enriched motifs for the modules that were created using the method 'weight>50.0%' (because these modules are not part
    # of the default settings of modules_from_adjacencies anymore.
    motifs = motifs[
        np.fromiter(map(compose(op.not_, contains('weight>50.0%')), motifs.Context), dtype=np.bool) & \
        np.fromiter(map(contains(*db_names), motifs.Context), dtype=np.bool) & \
        np.fromiter(map(contains('activating'), motifs.Context), dtype=np.bool)]

    # We build regulons only using enriched motifs with a NES of 3.0 or higher; we take only directly annotated TFs or TF annotated
    # for an orthologous gene into account; and we only keep regulons with at least 10 genes.
    regulons = list(filter(lambda r: len(r) >= 10, df2regulons(motifs[(motifs['NES'] >= 3.0) 
                                                                      & ((motifs['Annotation'] == 'gene is directly annotated')
                                                                        | (motifs['Annotation'].str.startswith('gene is orthologous to')
                                                                           & motifs['Annotation'].str.endswith('which is directly annotated for motif')))
                                                                     ])))
    
    # Rename regulons, i.e. remove suffix.
    return list(map(lambda r: r.rename(r.transcription_factor), regulons))


# In[28]:


regulons = derive_regulons(df_motifs)


# # AUCELL

# In[29]:


# Pickle these regulons.
with open('regulons.dat', 'wb') as f:
    pickle.dump(regulons, f)


# In[30]:


get_ipython().system("pyscenic aucell      {f_loom_path_scenic}      reg.csv      --output {'aucell.csv'}      --auc_threshold 0.10      --num_workers 10")


# In[31]:


regulons


# # now lets add scenic info to adata

# In[32]:


auc_mtx = pd.read_csv('aucell.csv', index_col=0)
auc_mtx


# In[33]:


add_scenic_metadata(adata_scn, auc_mtx, regulons)


# In[34]:


rss_cluster = regulon_specificity_scores( auc_mtx, adata_scn.obs.Tissue )
rss_cluster


# In[35]:


from adjustText import adjust_text


# In[36]:


# ============================================================
# RSS per Tissue/Subregion — clean, publication-friendly
#   - Works on older matplotlib (no supxlabel/supylabel)
#   - One row of panels
#   - adjustText tuned for small panels
#   - Optional per-panel y-min
#   - Saves SVG + PNG
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
from adjustText import adjust_text

# ----------------------------
# Inputs you already have:
#   - adata_scn.obs[label_key] : categories like Forebrain/Hindbrain/...
#   - rss_cluster : RSS matrix/data structure used by plot_rss()
#   - plot_rss(rss_cluster, cluster_name, top_n=..., max_n=None, ax=...)
# ----------------------------

label_key = "Tissue"   # <- change to "Subregion" if that's your column
out_svg = "Braun_et_al_week5pc_Tissues-RSS-top6.svg"
out_png = "Braun_et_al_week5pc_Tissues-RSS-top6.png"

# Choose categories in a stable order
if pd.api.types.is_categorical_dtype(adata_scn.obs[label_key]):
    cats = list(adata_scn.obs[label_key].cat.categories)
else:
    cats = sorted(adata_scn.obs[label_key].astype(str).unique())

# Optional: set a per-panel minimum y to avoid wasting vertical space
# (keys must match category names in `cats`)
ylim_min = {
    # "Forebrain": 0.48,
    # "Head": 0.35,
    # "Hindbrain": 0.43,
    # "Mesencephalon": 0.20,
    # "Midbrain": 0.20,
}

# How many labels per panel
TOP_N = 6   # try 6; bump to 8 only if still readable

# ----------------------------
# Figure layout: 1 row panels
# ----------------------------
n = len(cats)
fig_w = max(14, 4.8 * n)
fig, axes = plt.subplots(1, n, figsize=(fig_w, 4.6), constrained_layout=True)
if n == 1:
    axes = [axes]

for ax, c in zip(axes, cats):
    # Draw RSS plot
    plot_rss(rss_cluster, c, top_n=TOP_N, max_n=None, ax=ax)

    # Tighten y-lims (optional)
    try:
        x = rss_cluster.T[c]  # if rss_cluster is a DataFrame-like
        ymax = float(np.max(x))
        ymin = float(np.min(x))
    except Exception:
        ymax = None
        ymin = None

    if c in ylim_min and ymax is not None and ymin is not None:
        ax.set_ylim(ylim_min[c], ymax + (ymax - ymin) * 0.10)

    # Style text labels BEFORE adjustText
    for t in ax.texts:
        t.set_fontsize(10)

    # Improve label placement
    adjust_text(
        ax.texts,
        ax=ax,
        expand_text=(1.4, 1.6),
        expand_points=(1.2, 1.4),
        force_text=(1.0, 1.2),
        force_points=(0.3, 0.5),
        lim=300,
        only_move={"text": "xy"},
        arrowprops=dict(arrowstyle="-", lw=0.6, color="0.7"),
    )

    # Clean axes
    ax.set_title(str(c), fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("")

# Global labels (matplotlib-version safe)
fig.text(
    0.5, 0.02, "Regulon",
    ha="center", va="center", fontsize=12
)
fig.text(
    0.015, 0.5, "Regulon specificity score (RSS)",
    ha="center", va="center", rotation="vertical", fontsize=12
)

# Save
plt.savefig(out_svg, bbox_inches="tight")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.show()

print("Saved:", out_svg)
print("Saved:", out_png)


# In[38]:


adata_scn.write('adata_scenic_grn_braun_pc5.h5ad')


# ## start Cell Oracle now
# 

# In[1]:


import celloracle as co
co.check_python_requirements()


# In[2]:


import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
#os.environ["NUMBA_NUM_THREADS"] = "1"

import os, glob, re, pickle
import scanpy as sc
import pandas as pd
import numpy as np
import scanpy.external as sce
import re
from sklearn.metrics import pairwise_distances
import seaborn as sns
import matplotlib.pyplot as pl
import matplotlib.colors as colors
import operator as op
import anndata as ad
import loompy as lp
import celloracle as co
import pyarrow
#import scvelo as scv


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', None)
sc.set_figure_params(dpi = 150, dpi_save = 150, format = 'png')
sc._settings.ScanpyConfig(verbosity=0)


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

np.random.seed(44)


# In[ ]:





# In[3]:


wdir = "/mnt/nfs/CX000008_DS1/projects/gaurav/mistr/published_dataset_analyis/linnarsson_hbd_scenic/"
os.chdir( wdir )
adata_scn = sc.read_h5ad('adata_scenic_grn_braun_pc5.h5ad')
adata_scn


# In[4]:


sc.pp.subsample(adata_scn, n_obs= 25000)
adata_scn


# 

# In[5]:


######### read the adj file and look for TF targets
######### select targest above mean expression of importance 
adjacencies = pd.read_csv("adj.csv", index_col=False)
value = np.quantile(adjacencies['importance'].copy(), q = 0.99)
adj2 = adjacencies.copy()
adj2 = adjacencies[adjacencies['importance']>= value]
adj2


# ## here we select the tf and targets with importance above 0.99 quantile 
# - rearrange the tf and targets list 

# In[17]:


adj2.groupby('TF')['target'].agg(', '.join).reset_index().head(1000)
adj2.query("TF == 'GLI3'")


# In[6]:


adj3 = adj2.groupby('TF')['target'].agg(', '.join).reset_index().head(1000)
adj3


# ## time for cell oracle

# In[7]:


# Load base grn
base_GRN = co.data.load_human_promoter_base_GRN()

# Check data
base_GRN.head()



# In[9]:


# Make dictionary: dictionary key is TF and dictionary value is list of target genes.
TF_to_TG_dictionary = {}

for TF, TGs in zip(adj3.TF, adj3.target):
    # convert target gene to list
    TG_list = TGs.replace(" ", "").split(",")
    # store target gene list in a dictionary
    TF_to_TG_dictionary[TF] = TG_list

# We invert the dictionary above using a utility function in celloracle.
TG_to_TF_dictionary = co.utility.inverse_dictionary(TF_to_TG_dictionary)


# In[10]:


# we make a copy of adata for further analysis 
adata2 = adata_scn.copy()


# In[11]:


# Show data name in anndata
print("metadata columns :", list(adata2.obs.columns))
print("dimensional reduction: ", list(adata2.obsm.keys()))


# In[12]:


adata2


# In[13]:


# In this notebook, we use the unscaled mRNA count for the nput of Oracle object.
# N.B it will gibe a warning WARNING: adata.X seems to be already log-transformed. because there is decimal but the matrix is raw
adata2.X = adata2.layers["counts"].copy()

# Instantiate Oracle object
## i will name it with the gene i would like to purturb 
oracle = co.Oracle()
oracle.import_anndata_as_raw_count(adata=adata2,
                                   cluster_column_name="Tissue",
                                   embedding_name="X_umap")


# In[14]:


# You can load TF info dataframe with the following code.
oracle.import_TF_data(TF_info_matrix=base_GRN)

oracle.addTFinfo_dictionary(TG_to_TF_dictionary)


# In[15]:


# Perform PCA
oracle.perform_PCA()

# Select important PCs
plt.plot(np.cumsum(oracle.pca.explained_variance_ratio_)[:100])
n_comps = np.where(np.diff(np.diff(np.cumsum(oracle.pca.explained_variance_ratio_))>0.002))[0][0]
plt.axvline(n_comps, c="k")
print(n_comps)
n_comps = min(n_comps, 50)


# In[16]:


n_cell = oracle.adata.shape[0]
print(f"cell number is :{n_cell}")


# In[17]:


k = int(0.025*n_cell)
print(f"Auto-selected k is :{k}")


# In[18]:


oracle.knn_imputation(n_pca_dims=n_comps, k=k, balanced=True, b_sight=k*8,
                      b_maxl=k*4, n_jobs=4)


# ## time for GRN Calculation##

# In[19]:


sc.pl.umap(oracle.adata, color=['Tissue',"Age"], ncols = 3,wspace= 0.5, frameon= False)


# In[20]:


get_ipython().run_cell_magic('time', '', '# Calculate GRN for each population in "louvain_annot" clustering unit.\n# This step may take some time.(~30 minutes)\nlinks = oracle.get_links(cluster_name_for_GRN_unit="Tissue", alpha=10,\n                         verbose_level=10)\n')


# In[21]:


links.links_dict.keys()


# In[22]:


df = links.links_dict["Forebrain"]
mef2c_edges = df[df["source"] == "MEF2C"].sort_values(["-logp", "coef_abs"], ascending=False)
mef2c_edges.head(20)


# In[23]:


links.palette


# ## filter uncertain networks and keep top 2000 edges
# 
# ## time to calculate network scores 

# In[24]:


links.filter_links(p=0.001, weight="coef_abs", threshold_number=100)
links.plot_degree_distributions(plot_model=True)


# In[ ]:





# In[25]:


links.get_network_score()

links.merged_score.head(20)


# In[ ]:





# ## Network analysis; Network score for each gene

# In[26]:


# Check cluster name
links.cluster


# In[27]:


plt.rcParams["figure.figsize"] = [4, 6]

# Visualize top n-th genes with high scores.
links.plot_scores_as_rank(cluster="Forebrain", n_gene=20)


# In[28]:


cluster_name = "Forebrain"
filtered_links_df = links.filtered_links[cluster_name]
filtered_links_df.head()


# In[ ]:





# ## in silico perturbation

# In[29]:


oracle


# ## Make predictive models for simulation

# In[30]:


links.filter_links()
oracle.get_cluster_specific_TFdict_from_Links(links_object=links)
oracle.fit_GRN_for_simulation(alpha=10, use_cluster_specific_TFdict=True)


# In[31]:


plt.rcParams["figure.figsize"] = [4, 4]


# 

# # # simulate KO of GBX2

# In[32]:


# Check gene expression
goi = "GBX2"
sc.pl.umap(oracle.adata, color=[goi, oracle.cluster_column_name],
                 layer="imputed_count", use_raw=False, cmap="viridis")


# In[33]:


# Plot gene expression in histogram
sc.get.obs_df(oracle.adata, keys=[goi], layer="imputed_count").hist()
plt.show()


# In[34]:


# i will create a copy of oracle object to gene i want to knock out (GBX2)
oracle_GBX2 = oracle.copy()


# In[35]:


oracle_GBX2.simulate_shift(perturb_condition={goi: 0.0},
                      n_propagation=3)


# In[41]:


oracle_GBX2.simulate_shift(perturb_condition={goi: 0.0},
                      n_propagation=3)


# In[36]:


# Get transition probability
oracle_GBX2.estimate_transition_prob(n_neighbors=200,
                                knn_random=True,
                                sampled_fraction=1)

# Calculate embedding
oracle_GBX2.calculate_embedding_shift(sigma_corr=0.05)


# In[37]:


# Calculate embedding
oracle_GBX2.calculate_embedding_shift(sigma_corr=0.05)



fig, ax = plt.subplots(1, 2,  figsize=[13, 6])

scale =40
# Show quiver plot
oracle_GBX2.plot_quiver(scale=scale, ax=ax[0])
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

# Show quiver plot that was calculated with randomized graph.
oracle_GBX2.plot_quiver_random(scale=scale, ax=ax[1])
ax[1].set_title(f"Randomized simulation vector")

plt.show()


# In[38]:


# n_grid = 40 is a good starting value.
n_grid = 40
oracle_GBX2.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)
# Search for best min_mass.
oracle_GBX2.suggest_mass_thresholds(n_suggestion=12)


# In[39]:


min_mass = 78
oracle_GBX2.calculate_mass_filter(min_mass=min_mass, plot=True)


# In[40]:


fig, ax = plt.subplots(1, 2, figsize=[13, 6])

scale_simulation = 20

# Simulated KO flow
oracle_GBX2.plot_simulation_flow_on_grid(
    scale=scale_simulation,
    ax=ax[0]
)
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

# Randomized control flow
oracle_GBX2.plot_simulation_flow_random_on_grid(
    scale=scale_simulation,
    ax=ax[1]
)
ax[1].set_title("Randomized simulation vector")

# Save figure
out_png = f"{goi}_KO_simulation_flow.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")

plt.show()


# In[51]:


get_ipython().system('pwd')


# # # # simulate KO of MSX1 

# In[41]:


# Check gene expression
goi = "MSX1"
sc.pl.umap(oracle.adata, color=[goi, oracle.cluster_column_name],
                 layer="imputed_count", use_raw=False, cmap="viridis")


# In[42]:


# i will create a copy of oracle object to gene i want to knock out (MSX1)
oracle_MSX1 = oracle.copy()

oracle_MSX1.simulate_shift(perturb_condition={goi: 0.0},
                      n_propagation=3)


# Get transition probability
oracle_MSX1.estimate_transition_prob(n_neighbors=200,
                                knn_random=True,
                                sampled_fraction=1)

# Calculate embedding
oracle_MSX1.calculate_embedding_shift(sigma_corr=0.05)

# Calculate embedding
oracle_MSX1.calculate_embedding_shift(sigma_corr=0.05)



fig, ax = plt.subplots(1, 2,  figsize=[13, 6])

scale =40
# Show quiver plot
oracle_MSX1.plot_quiver(scale=scale, ax=ax[0])
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

# Show quiver plot that was calculated with randomized graph.
oracle_MSX1.plot_quiver_random(scale=scale, ax=ax[1])
ax[1].set_title(f"Randomized simulation vector")

plt.show()


# In[43]:


# n_grid = 40 is a good starting value.
n_grid = 40
oracle_MSX1.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)

min_mass = 78
oracle_MSX1.calculate_mass_filter(min_mass=min_mass, plot=True)


# In[44]:


fig, ax = plt.subplots(1, 2, figsize=[13, 6])

scale_simulation = 20

# Simulated KO flow
oracle_MSX1.plot_simulation_flow_on_grid(
    scale=scale_simulation,
    ax=ax[0]
)
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

# Randomized control flow
oracle_MSX1.plot_simulation_flow_random_on_grid(
    scale=scale_simulation,
    ax=ax[1]
)
ax[1].set_title("Randomized simulation vector")

# Save figure
out_png = f"{goi}_KO_simulation_flow.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")

plt.show()


# # # # simulate KO of PAX8

# 

# In[45]:


# Check gene expression
goi = "PAX8"
sc.pl.umap(oracle.adata, color=[goi, oracle.cluster_column_name],
                 layer="imputed_count", use_raw=False, cmap="viridis")


# In[46]:


# i will create a copy of oracle object to gene i want to knock out (MSX1)
oracle_PAX8 = oracle.copy()

oracle_PAX8.simulate_shift(perturb_condition={goi: 0.0},
                      n_propagation=3)


# Get transition probability
oracle_PAX8.estimate_transition_prob(n_neighbors=200,
                                knn_random=True,
                                sampled_fraction=1)

# Calculate embedding
oracle_PAX8.calculate_embedding_shift(sigma_corr=0.05)

# Calculate embedding
oracle_PAX8.calculate_embedding_shift(sigma_corr=0.05)



fig, ax = plt.subplots(1, 2,  figsize=[13, 6])

scale =40
# Show quiver plot
oracle_PAX8.plot_quiver(scale=scale, ax=ax[0])
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

# Show quiver plot that was calculated with randomized graph.
oracle_PAX8.plot_quiver_random(scale=scale, ax=ax[1])
ax[1].set_title(f"Randomized simulation vector")

plt.show()


# In[49]:


# n_grid = 40 is a good starting value.
n_grid = 40
oracle_PAX8.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)

min_mass = 78
oracle_PAX8.calculate_mass_filter(min_mass=min_mass, plot=True)


# In[50]:


fig, ax = plt.subplots(1, 2, figsize=[13, 6])

scale_simulation = 20

# Simulated KO flow
oracle_PAX8.plot_simulation_flow_on_grid(
    scale=scale_simulation,
    ax=ax[0]
)
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

# Randomized control flow
oracle_PAX8.plot_simulation_flow_random_on_grid(
    scale=scale_simulation,
    ax=ax[1]
)
ax[1].set_title("Randomized simulation vector")

# Save figure
out_png = f"{goi}_KO_simulation_flow.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")

plt.show()


# # # # simulate KO of OTX2

# In[51]:


# Check gene expression
goi = "OTX2"
sc.pl.umap(oracle.adata, color=[goi, oracle.cluster_column_name],
                 layer="imputed_count", use_raw=False, cmap="viridis")


# In[52]:


# i will create a copy of oracle object to gene i want to knock out (MSX1)
oracle_OTX2 = oracle.copy()

oracle_OTX2.simulate_shift(perturb_condition={goi: 0.0},
                      n_propagation=3)


# Get transition probability
oracle_OTX2.estimate_transition_prob(n_neighbors=200,
                                knn_random=True,
                                sampled_fraction=1)

# Calculate embedding
oracle_OTX2.calculate_embedding_shift(sigma_corr=0.05)

# Calculate embedding
oracle_OTX2.calculate_embedding_shift(sigma_corr=0.05)



fig, ax = plt.subplots(1, 2,  figsize=[13, 6])

scale =40
# Show quiver plot
oracle_OTX2.plot_quiver(scale=scale, ax=ax[0])
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

# Show quiver plot that was calculated with randomized graph.
oracle_OTX2.plot_quiver_random(scale=scale, ax=ax[1])
ax[1].set_title(f"Randomized simulation vector")

plt.show()


# In[54]:


# n_grid = 40 is a good starting value.
n_grid = 40
oracle_OTX2.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)

min_mass = 78
oracle_OTX2.calculate_mass_filter(min_mass=min_mass, plot=True)


# In[55]:


fig, ax = plt.subplots(1, 2, figsize=[13, 6])

scale_simulation = 20

# Simulated KO flow
oracle_OTX2.plot_simulation_flow_on_grid(
    scale=scale_simulation,
    ax=ax[0]
)
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

# Randomized control flow
oracle_OTX2.plot_simulation_flow_random_on_grid(
    scale=scale_simulation,
    ax=ax[1]
)
ax[1].set_title("Randomized simulation vector")

# Save figure
out_png = f"{goi}_KO_simulation_flow.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")

plt.show()


# # # # simulate KO of PAX5

# In[56]:


# Check gene expression
goi = "PAX5"
sc.pl.umap(oracle.adata, color=[goi, oracle.cluster_column_name],
                 layer="imputed_count", use_raw=False, cmap="viridis")


# In[57]:


# i will create a copy of oracle object to gene i want to knock out (MSX1)
oracle_PAX5 = oracle.copy()

oracle_PAX5.simulate_shift(perturb_condition={goi: 0.0},
                      n_propagation=3)


# Get transition probability
oracle_PAX5.estimate_transition_prob(n_neighbors=200,
                                knn_random=True,
                                sampled_fraction=1)

# Calculate embedding
oracle_PAX5.calculate_embedding_shift(sigma_corr=0.05)

# Calculate embedding
oracle_PAX5.calculate_embedding_shift(sigma_corr=0.05)



fig, ax = plt.subplots(1, 2,  figsize=[13, 6])

scale =40
# Show quiver plot
oracle_PAX5.plot_quiver(scale=scale, ax=ax[0])
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

# Show quiver plot that was calculated with randomized graph.
oracle_PAX5.plot_quiver_random(scale=scale, ax=ax[1])
ax[1].set_title(f"Randomized simulation vector")

plt.show()


# In[58]:


# n_grid = 40 is a good starting value.
n_grid = 40
oracle_PAX5.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)

min_mass = 78
oracle_PAX5.calculate_mass_filter(min_mass=min_mass, plot=True)


# In[59]:


fig, ax = plt.subplots(1, 2, figsize=[13, 6])

scale_simulation = 20

# Simulated KO flow
oracle_PAX5.plot_simulation_flow_on_grid(
    scale=scale_simulation,
    ax=ax[0]
)
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

# Randomized control flow
oracle_PAX5.plot_simulation_flow_random_on_grid(
    scale=scale_simulation,
    ax=ax[1]
)
ax[1].set_title("Randomized simulation vector")

# Save figure
out_png = f"{goi}_KO_simulation_flow.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")

plt.show()


# # # # simulate KO of PAX2

# In[60]:


# Check gene expression
goi = "PAX2"
sc.pl.umap(oracle.adata, color=[goi, oracle.cluster_column_name],
                 layer="imputed_count", use_raw=False, cmap="viridis")


# In[61]:


# i will create a copy of oracle object to gene i want to knock out (MSX1)
oracle_PAX2 = oracle.copy()

oracle_PAX2.simulate_shift(perturb_condition={goi: 0.0},
                      n_propagation=3)


# Get transition probability
oracle_PAX2.estimate_transition_prob(n_neighbors=200,
                                knn_random=True,
                                sampled_fraction=1)

# Calculate embedding
oracle_PAX2.calculate_embedding_shift(sigma_corr=0.05)

# Calculate embedding
oracle_PAX2.calculate_embedding_shift(sigma_corr=0.05)



fig, ax = plt.subplots(1, 2,  figsize=[13, 6])

scale =40
# Show quiver plot
oracle_PAX2.plot_quiver(scale=scale, ax=ax[0])
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

# Show quiver plot that was calculated with randomized graph.
oracle_PAX2.plot_quiver_random(scale=scale, ax=ax[1])
ax[1].set_title(f"Randomized simulation vector")

plt.show()


# In[62]:


# n_grid = 40 is a good starting value.
n_grid = 40
oracle_PAX2.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)

min_mass = 78
oracle_PAX2.calculate_mass_filter(min_mass=min_mass, plot=True)


# In[63]:


fig, ax = plt.subplots(1, 2, figsize=[13, 6])

scale_simulation = 20

# Simulated KO flow
oracle_PAX2.plot_simulation_flow_on_grid(
    scale=scale_simulation,
    ax=ax[0]
)
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

# Randomized control flow
oracle_PAX2.plot_simulation_flow_random_on_grid(
    scale=scale_simulation,
    ax=ax[1]
)
ax[1].set_title("Randomized simulation vector")

# Save figure
out_png = f"{goi}_KO_simulation_flow.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")

plt.show()


# # # # simulate KO of EN1

# In[64]:


# Check gene expression
goi = "EN1"
sc.pl.umap(oracle.adata, color=[goi, oracle.cluster_column_name],
                 layer="imputed_count", use_raw=False, cmap="viridis")


# In[65]:


# i will create a copy of oracle object to gene i want to knock out (MSX1)
oracle_EN1 = oracle.copy()

oracle_EN1.simulate_shift(perturb_condition={goi: 0.0},
                      n_propagation=3)


# Get transition probability
oracle_EN1.estimate_transition_prob(n_neighbors=200,
                                knn_random=True,
                                sampled_fraction=1)

# Calculate embedding
oracle_EN1.calculate_embedding_shift(sigma_corr=0.05)

# Calculate embedding
oracle_EN1.calculate_embedding_shift(sigma_corr=0.05)



fig, ax = plt.subplots(1, 2,  figsize=[13, 6])

scale =40
# Show quiver plot
oracle_EN1.plot_quiver(scale=scale, ax=ax[0])
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

# Show quiver plot that was calculated with randomized graph.
oracle_EN1.plot_quiver_random(scale=scale, ax=ax[1])
ax[1].set_title(f"Randomized simulation vector")

plt.show()


# In[66]:


# n_grid = 40 is a good starting value.
n_grid = 40
oracle_EN1.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)

min_mass = 78
oracle_EN1.calculate_mass_filter(min_mass=min_mass, plot=True)


# In[67]:


fig, ax = plt.subplots(1, 2, figsize=[13, 6])

scale_simulation = 20

# Simulated KO flow
oracle_EN1.plot_simulation_flow_on_grid(
    scale=scale_simulation,
    ax=ax[0]
)
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

# Randomized control flow
oracle_EN1.plot_simulation_flow_random_on_grid(
    scale=scale_simulation,
    ax=ax[1]
)
ax[1].set_title("Randomized simulation vector")

# Save figure
out_png = f"{goi}_KO_simulation_flow.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")

plt.show()

