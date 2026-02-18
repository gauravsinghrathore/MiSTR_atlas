#!/usr/bin/env python
# coding: utf-8

# In[1]:


#############load necessary libraries##############

import os
os.environ["MKL_NUM_THREADS"] = "30"
os.environ["NUMEXPR_NUM_THREADS"] = "30"
os.environ["OMP_NUM_THREADS"] = "30"

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
import scvelo as scv





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


# In[2]:


#############define necessary functions##############

def Get_genes_to_keep(loaded_obj):
    detected_genes = loaded_obj[:,sc.pp.filter_genes(loaded_obj, min_cells=3, 
                                                     inplace=False)[0]].var_names.tolist()
    MitoRiboGenes = (list(filter(lambda x: re.search(r'^Mt-', x, re.IGNORECASE), detected_genes)) + 
                     list(filter(lambda x: re.search(r'^Mtrnr', x, re.IGNORECASE), detected_genes)) + 
                     list(filter(lambda x: re.search(r'^Rpl', x, re.IGNORECASE), detected_genes)) +
                     # "^RP[0-5][0-9]" catches a lot of non ribosomal proteins like LINCs, Pseudogenes and even prot coding genes
                     #list(filter(lambda x: re.search(r'^RP[0-5][0-9]', x, re.IGNORECASE), detected_genes)) +
                     list(filter(lambda x: re.search(r'^Rps', x, re.IGNORECASE), detected_genes)))
    return(list(set(detected_genes) - set(MitoRiboGenes)))


def PreProcessing(obj, QCplots=False, Verbosity="high"):
    obj = obj
  
    if QCplots:
        sc.pl.violin(obj, ['percent_mito', 'percent_ribo', 'n_counts'], jitter=0.4, multi_panel=True)

    # first round of gene filtering
    Genes_to_keep = Get_genes_to_keep(obj)
    obj = obj[:,Genes_to_keep]
    if Verbosity=="high":
        print("First gene filtering: "+str(obj.X.shape))

    # Cells filtering
    sc.pp.filter_cells(obj, min_genes=200)
    median_genes_per_cell = np.median(obj.X.getnnz(axis=1)) # recomputing this per each dataset will introduce inconsistent thrholds
    sc.pp.filter_cells(obj, max_genes=2.5*median_genes_per_cell)
    if Verbosity=="high":
        print("Cell filtering: "+str(obj.X.shape))

    # Second round of gene filtering: some genes got to all-zeros after cell filtering
    Genes_to_keep_2 = Get_genes_to_keep(obj)
    obj = obj[:,Genes_to_keep_2]
    if Verbosity=="high":
        print("Second gene filtering: "+str(obj.X.shape))

    obj = obj
    if QCplots:
        sc.pl.violin(obj, ['percent_mito', 'percent_ribo', 
                           'n_counts', 'n_genes'], jitter=0.4, multi_panel=True)

    if Verbosity=="high":
        print("Raw!")
        #print(obj.X[:10,:10])

    sc.pp.normalize_total(obj, layers="all")
    if Verbosity=="high":
        print("Normalized")
        #print(obj.X[:10,:10])

    sc.pp.log1p(obj)
    if Verbosity=="high":
        print("Log transformed")
        #print(obj.X[:10,:10])

    sc.pp.highly_variable_genes(obj) # needs log transformed data
    if QCplots:
        sc.pl.highly_variable_genes(obj)
    #if Verbosity=="high":
        #print("HVG subset: "+str(obj.shape))
    
    return(obj)


# PROVIDE CELL CYCLE GENE LISTS FOR PHASE SCORING
S_genes_file = "/home/kgr851/new_analysis_new_cc/cc/s.txt"
G2M_genes_file = "/home/kgr851/new_analysis_new_cc/cc/g2m.txt"
S_phase_genes = [x.strip() for x in open(S_genes_file)]
G2M_phase_genes = [x.strip() for x in open(S_genes_file)]

def CellCycleRegression(obj):
    
    # N.B. Regression is only performed on matrix .X
    
        print("Regressing individual phases")
        sc.pp.scale(obj)
        sc.tl.score_genes_cell_cycle(obj, s_genes=S_phase_genes, g2m_genes=G2M_phase_genes)
        sc.pp.regress_out(obj, ['S_score', 'G2M_score'])
        sc.pp.scale(obj)
        # Store "old_phase" for later plotting of regression effect on phase%
        obj.obs['OldPhase'] = obj.obs['phase']
        sc.tl.score_genes_cell_cycle(obj, s_genes=S_phase_genes, g2m_genes=G2M_phase_genes)
        print("Regressed")
        print(obj.X[:10,:10])
   
        return(obj)

def subset_two_genes2(adata, gene1: str ,gene2: str, coexpressed: bool = False):
    gene1_cells = adata[adata[: , gene1].X > 0.8, :].obs_names
    gene2_cells = adata[adata[: , gene2].X > 0.8, :].obs_names
    
    if coexpressed:
        cells = np.intersect1d(gene1_cells, gene2_cells)
    else:
         cells = np.unique(np.concatenate([gene1_cells, gene2_cells]))
    
    return adata[cells, ].copy()

# def subset_two_genes(adata, gene1: str ,gene2: str):
#     x = adata[adata[: , gene1].X > 0.75, :]
#     y = adata[adata[: , gene2].X > 0.75, :]
#     common_obs = np.intersect1d(x.obs_names, y.obs_names)
#     mix = adata[common_obs]
#     x = x[~x.obs_names.isin(common_obs)].copy()
#     y = y[~y.obs_names.isin(common_obs)].copy()
    
# print('subset data is ready')
    
# subset = mix.concatenate(x,y, batch_key = False)
    
# return subset
    

# def subset_two_genes2(adata, gene1: str ,gene2: str, coexpressed: bool = False):
    
#     value1 = np.mean(adata[:, gene1].X.toarray().flatten())+np.std(adata[:, gene1].X.toarray().flatten())
#     value2 = np.mean(adata[:, gene2].X.toarray().flatten())+np.std(adata[:, gene2].X.toarray().flatten())
    
#     gene1_cells = adata[adata[: , gene1].X > value1, :].obs_names
#     gene2_cells = adata[adata[: , gene2].X > value2, :].obs_names
    
#     if coexpressed:
#         cells = np.intersect1d(gene1_cells, gene2_cells)
#     else:
#         cells = np.unique(np.concatenate([gene1_cells, gene2_cells]))
    
#     return adata[cells, ].copy()


# In[3]:


def clean_read(dataset: str) -> sc.AnnData:
    adata = sc.read_h5ad(dataset)
    
    # shorten metadata
   # adata.obs = adata.obs[['seurat_clusters']]
    del adata.obsm
    del adata.layers
    
    return adata


# # load adata for day 5 to 14 cells expressing FOXA2 or NKx2-1 

# In[4]:


all_mistr_nf = sc.read_h5ad('mistr_day5-14_nf.h5ad')
all_mistr_nf


# In[5]:


###### subset OTX2+ cells only######
all_mistr_nf_otx2 = all_mistr_nf[all_mistr_nf[: , 'OTX2'].X > 0, :]
all_mistr_nf_otx2


# In[6]:


sc.pl.draw_graph(all_mistr_nf_otx2, color=['FOXA2','NKX2-1','OTX2','cluster','day'],color_map='viridis',frameon=False, vmax = 3)


# In[8]:


adata  = all_mistr_nf_otx2.copy()
adata.X = adata.layers["counts"].copy()
### select top 3000 hvg for analysis #####
sc.pp.highly_variable_genes(
    adata,
    flavor="seurat_v3",
    n_top_genes=3_000,
    subset=True
)


# In[9]:


adata.uns['cluster_colors']


# In[10]:


adata.uns['cluster_colors'][0] = '#ff0000'
adata.uns['cluster_colors'][1] = '#00ff00'
adata.uns['cluster_colors'][2] = '#FFFF00'


# In[ ]:





# In[12]:


with plt.rc_context({"figure.figsize": [5, 5]}):
    sc.pl.draw_graph(adata, color=['cluster'],size = 20,frameon=False,save = "NKX_FOXA2_subset_OTX2_cells.svg" ,
                     vmax = 4, sort_order= False)


# In[13]:


with plt.rc_context({"figure.figsize": [5, 5]}):
    sc.pl.draw_graph(adata, color=['HTO_classification'],size = 20,frameon=False,save = "NKX_FOXA2_subset_OTX2_HTO_classification.svg" ,
                     vmax = 4, sort_order= False)


# ## here we read the tf and targets from grn boost 

# In[16]:


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
adj2.query("TF == 'NKX2-1'")


# In[18]:


adj3 = adj2.groupby('TF')['target'].agg(', '.join).reset_index().head(1000)
adj3


# # time for cell oracle 

# In[19]:


# Load base grn
base_GRN = co.data.load_human_promoter_base_GRN()

# Check data
base_GRN.head()


# In[20]:


# Make dictionary: dictionary key is TF and dictionary value is list of target genes.
TF_to_TG_dictionary = {}

for TF, TGs in zip(adj3.TF, adj3.target):
    # convert target gene to list
    TG_list = TGs.replace(" ", "").split(",")
    # store target gene list in a dictionary
    TF_to_TG_dictionary[TF] = TG_list

# We invert the dictionary above using a utility function in celloracle.
TG_to_TF_dictionary = co.utility.inverse_dictionary(TF_to_TG_dictionary)


# In[21]:


# we make a copy of adata for further analysis 
adata2 = adata.copy()


# In[22]:


# Show data name in anndata
print("metadata columns :", list(adata2.obs.columns))
print("dimensional reduction: ", list(adata2.obsm.keys()))


# In[ ]:





# In[23]:


# In this notebook, we use the unscaled mRNA count for the nput of Oracle object.
# N.B it will gibe a warning WARNING: adata.X seems to be already log-transformed. because there is decimal but the matrix is raw
adata2.X = adata2.layers["counts"].copy()

# Instantiate Oracle object
## i will name it with the gene i would like to purturb 
oracle = co.Oracle()
oracle.import_anndata_as_raw_count(adata=adata,
                                   cluster_column_name="cluster",
                                   embedding_name="X_draw_graph_fa")


# In[24]:


# cross confirm
adata.layers["counts"].todense()


# In[25]:


# You can load TF info dataframe with the following code.
oracle.import_TF_data(TF_info_matrix=base_GRN)


# In[26]:


oracle.addTFinfo_dictionary(TG_to_TF_dictionary)


# In[27]:


# Perform PCA
oracle.perform_PCA()

# Select important PCs
plt.plot(np.cumsum(oracle.pca.explained_variance_ratio_)[:100])
n_comps = np.where(np.diff(np.diff(np.cumsum(oracle.pca.explained_variance_ratio_))>0.002))[0][0]
plt.axvline(n_comps, c="k")
print(n_comps)
n_comps = min(n_comps, 50)


# In[28]:


n_cell = oracle.adata.shape[0]
print(f"cell number is :{n_cell}")


# In[29]:


k = int(0.025*n_cell)
print(f"Auto-selected k is :{k}")


# In[30]:


oracle.knn_imputation(n_pca_dims=n_comps, k=k, balanced=True, b_sight=k*8,
                      b_maxl=k*4, n_jobs=4)


# In[ ]:





# 
# ## time for GRN Calculation##

# In[31]:


sc.pl.draw_graph(oracle.adata, color=['cluster',"day"], ncols = 3,wspace= 0.5, frameon= False)


# In[32]:


get_ipython().run_cell_magic('time', '', '# Calculate GRN for each population in "louvain_annot" clustering unit.\n# This step may take some time.(~30 minutes)\nlinks = oracle.get_links(cluster_name_for_GRN_unit="cluster", alpha=10,\n                         verbose_level=10)\n')


# In[33]:


links.links_dict.keys()


# In[34]:


links.links_dict['NKX2-1']


# In[35]:


links.palette


# ## filter uncertain networks and keep top 2000 edges

# ## time to calculate network scores 

# In[37]:


links.get_network_score()


# In[38]:


links.merged_score.head(20)


# In[39]:


## Network analysis; Network score for each gene


# In[40]:


# Check cluster name
links.cluster


# In[41]:


plt.rcParams["figure.figsize"] = [4, 6]


# In[45]:


plt.rcParams["figure.figsize"] = [8, 6]


# In[47]:


cluster_name = "FOXA2"
filtered_links_df = links.filtered_links[cluster_name]
filtered_links_df.head()


# In[48]:


### save the oracle object and links#######
oracle.to_hdf5("mistr_foxa2_nkx2-1.celloracle.oracle")
links.to_hdf5(file_path="mistr_foxa2_nkx2-1.celloracle.links")


# # in silico perturbation 
# 

# In[49]:


oracle


# ## Make predictive models for simulation

# In[50]:


links.filter_links()
oracle.get_cluster_specific_TFdict_from_Links(links_object=links)
oracle.fit_GRN_for_simulation(alpha=10, use_cluster_specific_TFdict=True)


# In[51]:


plt.rcParams["figure.figsize"] = [4, 4]


# In[ ]:





# # IRX1 KO 

# In[129]:


# Check gene expression
goi = "IRX1"
sc.pl.draw_graph(oracle.adata, color=[goi, oracle.cluster_column_name],
                 layer="imputed_count", use_raw=False, cmap="viridis")


# In[130]:


# i will create a copy of oracle objext to gene i want to knock out (IRX1)
oracle_IRX1 = oracle.copy()
oracle_IRX1.simulate_shift(perturb_condition={goi: 0.0},
                      n_propagation=3)


# In[131]:


## visualise KO simulation
# Get transition probability
oracle_IRX1.estimate_transition_prob(n_neighbors=200,
                                knn_random=True,
                                sampled_fraction=1)

# Calculate embedding
oracle_IRX1.calculate_embedding_shift(sigma_corr=0.05)



fig, ax = plt.subplots(1, 2,  figsize=[13, 6])

scale =20
# Show quiver plot
oracle_IRX1.plot_quiver(scale=scale, ax=ax[0])
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

# Show quiver plot that was calculated with randomized graph.
oracle_IRX1.plot_quiver_random(scale=scale, ax=ax[1])
ax[1].set_title(f"Randomized simulation vector")

plt.show()


# In[132]:


# n_grid = 40 is a good starting value.
n_grid = 40
oracle_IRX1.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)
# Search for best min_mass.
oracle_IRX1.suggest_mass_thresholds(n_suggestion=12)


# In[133]:


min_mass = 0.0026
oracle_IRX1.calculate_mass_filter(min_mass=min_mass, plot=True)


# ## compare simulation vectors with development vectors

# In[136]:


from celloracle.applications import Gradient_calculator

# Instantiate Gradient calculator object
gradient = Gradient_calculator(oracle_object=oracle_IRX1, pseudotime_key="dpt_pseudotime")


# In[137]:


gradient.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)
gradient.calculate_mass_filter(min_mass=min_mass, plot=True)


# In[139]:


# Calculate graddient
gradient.calculate_gradient()

# Show results
scale_dev = 30
gradient.visualize_results(scale=scale_dev, s=5)



# In[140]:


from celloracle.applications import Oracle_development_module

# Make Oracle_development_module to compare two vector field
dev = Oracle_development_module()

# Load development flow
dev.load_differentiation_reference_data(gradient_object=gradient)

# Load simulation result
dev.load_perturb_simulation_data(oracle_object=oracle_IRX1)


# Calculate inner produc scores
dev.calculate_inner_product()
dev.calculate_digitized_ip(n_bins=10)


# In[141]:


# Let's visualize the results
dev.visualize_development_module_layout_0(s=5,
                                          scale_for_simulation=scale_simulation,
                                          s_grid=50,
                                          scale_for_pseudotime=scale_dev,
                                          vm=0.02)


# In[142]:


##### Show perturbation scores with perturbation simulation vector field
fig, ax = plt.subplots(figsize=[7, 7])
dev.plot_inner_product_on_grid(vm=0.005, s=50, ax=ax)
dev.plot_simulation_flow_on_grid(scale=0.15, show_background=False, ax=ax)
plt.savefig("co_plots/IRX1KO_score.png", dpi=150, bbox_inches = "tight")


# # SIX3 KO

# In[143]:


# Check gene expression
goi = "SIX3"
sc.pl.draw_graph(oracle.adata, color=[goi, oracle.cluster_column_name],
                 layer="imputed_count", use_raw=False, cmap="viridis")


# In[144]:


# i will create a copy of oracle objext to gene i want to knock out (SIX3)
oracle_SIX3 = oracle.copy()
oracle_SIX3.simulate_shift(perturb_condition={goi: 0.0},
                      n_propagation=3)


# In[145]:


## visualise KO simulation
# Get transition probability
oracle_SIX3.estimate_transition_prob(n_neighbors=200,
                                knn_random=True,
                                sampled_fraction=1)

# Calculate embedding
oracle_SIX3.calculate_embedding_shift(sigma_corr=0.05)



fig, ax = plt.subplots(1, 2,  figsize=[13, 6])

scale =50
# Show quiver plot
oracle_SIX3.plot_quiver(scale=scale, ax=ax[0])
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

# Show quiver plot that was calculated with randomized graph.
oracle_SIX3.plot_quiver_random(scale=scale, ax=ax[1])
ax[1].set_title(f"Randomized simulation vector")

plt.show()


# In[146]:


# n_grid = 40 is a good starting value.
n_grid = 40
oracle_SIX3.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)
# Search for best min_mass.
oracle_SIX3.suggest_mass_thresholds(n_suggestion=12)


# In[147]:


min_mass = 0.0026
oracle_SIX3.calculate_mass_filter(min_mass=min_mass, plot=True)


# ## compare simulation vectors with development vectors 

# In[150]:


from celloracle.applications import Gradient_calculator

# Instantiate Gradient calculator object
gradient = Gradient_calculator(oracle_object=oracle_SIX3, pseudotime_key="dpt_pseudotime")


# In[151]:


gradient.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)
gradient.calculate_mass_filter(min_mass=min_mass, plot=True)


# In[153]:


# Calculate graddient
gradient.calculate_gradient()

# Show results
scale_dev = 30
gradient.visualize_results(scale=scale_dev, s=5)



# In[154]:


from celloracle.applications import Oracle_development_module

# Make Oracle_development_module to compare two vector field
dev = Oracle_development_module()

# Load development flow
dev.load_differentiation_reference_data(gradient_object=gradient)

# Load simulation result
dev.load_perturb_simulation_data(oracle_object=oracle_SIX3)


# Calculate inner produc scores
dev.calculate_inner_product()
dev.calculate_digitized_ip(n_bins=10)


# In[155]:


# Let's visualize the results
dev.visualize_development_module_layout_0(s=5,
                                          scale_for_simulation=scale_simulation,
                                          s_grid=50,
                                          scale_for_pseudotime=scale_dev,
                                          vm=0.02)


# In[156]:


# Show perturbation scores with perturbation simulation vector field
fig, ax = plt.subplots(figsize=[7, 7])
dev.plot_inner_product_on_grid(vm=0.01, s=50, ax=ax)
dev.plot_simulation_flow_on_grid(scale=0.6, show_background=False, ax=ax)
plt.savefig("co_plots/SIX3KO_score.png", dpi=150, bbox_inches = "tight")


# # TCF7L2 KO 

# In[186]:


# Check gene expression
goi = "TCF7L2"
sc.pl.draw_graph(oracle.adata, color=[goi, oracle.cluster_column_name],
                 layer="imputed_count", use_raw=False, cmap="viridis")


# In[187]:


# i will create a copy of oracle objext to gene i want to knock out (TCF7L2)
oracle_TCF7L2 = oracle.copy()
oracle_TCF7L2.simulate_shift(perturb_condition={goi: 0.0},
                      n_propagation=3)


# In[188]:


## visualise KO simulation
# Get transition probability
oracle_TCF7L2.estimate_transition_prob(n_neighbors=200,
                                knn_random=True,
                                sampled_fraction=1)

# Calculate embedding
oracle_TCF7L2.calculate_embedding_shift(sigma_corr=0.05)



fig, ax = plt.subplots(1, 2,  figsize=[13, 6])

scale = 40
# Show quiver plot
oracle_TCF7L2.plot_quiver(scale=scale, ax=ax[0])
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

# Show quiver plot that was calculated with randomized graph.
oracle_TCF7L2.plot_quiver_random(scale=scale, ax=ax[1])
ax[1].set_title(f"Randomized simulation vector")

plt.show()


# In[189]:


# n_grid = 40 is a good starting value.
n_grid = 40
oracle_TCF7L2.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)
# Search for best min_mass.
oracle_TCF7L2.suggest_mass_thresholds(n_suggestion=12)


# In[190]:


min_mass = 0.0026
oracle_TCF7L2.calculate_mass_filter(min_mass=min_mass, plot=True)


# In[192]:


## compare simulation vectors with development vectors


# In[194]:


from celloracle.applications import Gradient_calculator

# Instantiate Gradient calculator object
gradient = Gradient_calculator(oracle_object=oracle_TCF7L2, pseudotime_key="dpt_pseudotime")


# In[195]:


gradient.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)
gradient.calculate_mass_filter(min_mass=min_mass, plot=True)


# In[197]:


# Calculate graddient
gradient.calculate_gradient()

# Show results
scale_dev = 30
gradient.visualize_results(scale=scale_dev, s=5)



# In[198]:


from celloracle.applications import Oracle_development_module

# Make Oracle_development_module to compare two vector field
dev = Oracle_development_module()

# Load development flow
dev.load_differentiation_reference_data(gradient_object=gradient)

# Load simulation result
dev.load_perturb_simulation_data(oracle_object=oracle_TCF7L2)


# Calculate inner produc scores
dev.calculate_inner_product()
dev.calculate_digitized_ip(n_bins=10)


# In[199]:


# Let's visualize the results
dev.visualize_development_module_layout_0(s=5,
                                          scale_for_simulation=scale_simulation,
                                          s_grid=50,
                                          scale_for_pseudotime=scale_dev,
                                          vm=0.02)


# In[200]:


# Show perturbation scores with perturbation simulation vector field
fig, ax = plt.subplots(figsize=[7, 7])
dev.plot_inner_product_on_grid(vm=0.01, s=50, ax=ax)
dev.plot_simulation_flow_on_grid(scale=0.7, show_background=False, ax=ax)
plt.savefig("co_plots/TCF7L2KO_score.png", dpi=150, bbox_inches = "tight")


# # RAX KO 

# In[201]:


# Check gene expression
goi = "RAX"
sc.pl.draw_graph(oracle.adata, color=[goi, oracle.cluster_column_name],
                 layer="imputed_count", use_raw=False, cmap="viridis")


# In[202]:


# i will create a copy of oracle objext to gene i want to knock out (RAX)
oracle_RAX = oracle.copy()
oracle_RAX.simulate_shift(perturb_condition={goi: 0.0},
                      n_propagation=3)


# In[203]:


## visualise KO simulation
# Get transition probability
oracle_RAX.estimate_transition_prob(n_neighbors=200,
                                knn_random=True,
                                sampled_fraction=1)

# Calculate embedding
oracle_RAX.calculate_embedding_shift(sigma_corr=0.05)



fig, ax = plt.subplots(1, 2,  figsize=[13, 6])

scale = 40
# Show quiver plot
oracle_RAX.plot_quiver(scale=scale, ax=ax[0])
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

# Show quiver plot that was calculated with randomized graph.
oracle_RAX.plot_quiver_random(scale=scale, ax=ax[1])
ax[1].set_title(f"Randomized simulation vector")

plt.show()


# In[204]:


# n_grid = 40 is a good starting value.
n_grid = 40
oracle_RAX.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)
# Search for best min_mass.
oracle_RAX.suggest_mass_thresholds(n_suggestion=12)


# In[205]:


min_mass = 0.0026
oracle_RAX.calculate_mass_filter(min_mass=min_mass, plot=True)


# In[207]:


## compare simulation vectors with development vectors


# In[209]:


from celloracle.applications import Gradient_calculator

# Instantiate Gradient calculator object
gradient = Gradient_calculator(oracle_object=oracle_RAX, pseudotime_key="dpt_pseudotime")


# In[210]:


gradient.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)
gradient.calculate_mass_filter(min_mass=min_mass, plot=True)


# In[212]:


# Calculate graddient
gradient.calculate_gradient()

# Show results
scale_dev = 30
gradient.visualize_results(scale=scale_dev, s=5)



# In[213]:


from celloracle.applications import Oracle_development_module

# Make Oracle_development_module to compare two vector field
dev = Oracle_development_module()

# Load development flow
dev.load_differentiation_reference_data(gradient_object=gradient)

# Load simulation result
dev.load_perturb_simulation_data(oracle_object=oracle_RAX)


# Calculate inner produc scores
dev.calculate_inner_product()
dev.calculate_digitized_ip(n_bins=10)


# In[216]:


# Let's visualize the results
dev.visualize_development_module_layout_0(s=5,
                                          scale_for_simulation=scale_simulation,
                                          s_grid=50,
                                          scale_for_pseudotime=scale_dev,
                                          vm=0.02)
plt.savefig("co_plots/RAXKO_details.png", dpi=500, bbox_inches = "tight")


# In[215]:


# Show perturbation scores with perturbation simulation vector field
fig, ax = plt.subplots(figsize=[7, 7])
dev.plot_inner_product_on_grid(vm=0.01, s=50, ax=ax)
dev.plot_simulation_flow_on_grid(scale=0.5, show_background=False, ax=ax)
plt.savefig("co_plots/RAXKO_score.png", bbox_inches = "tight")


# In[218]:


get_ipython().system('jupyter nbconvert grn+NKX2-1+FOXA2+OTX2+COv.ipynb --to html')


# In[1]:


get_ipython().system('jupyter nbconvert --to script your_notebook.ipynb')

