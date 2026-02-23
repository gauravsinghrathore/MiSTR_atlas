#!/usr/bin/env python
# coding: utf-8

# In[3]:


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
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import operator as op
import anndata as ad
import loompy as lp
import celloracle as co
import pyarrow
plt.rcParams["savefig.dpi"] = 600





get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', None)
sc.set_figure_params(dpi = 150, dpi_save = 150, format = 'png')
sc._settings.ScanpyConfig(verbosity=0)


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


# # read the mhb subsetted adata

# In[4]:


mhb_mistr = sc.read_h5ad('mhb_all.h5ad')
mhb_mistr


# In[5]:


sc.pl.draw_graph(mhb_mistr, color=['FGF17','FGF8','PAX2','PAX8','EN1','WNT1','louvain'],color_map='viridis',frameon=False)


# In[6]:


adata  = mhb_mistr.copy()

adata.X = adata.layers['counts'].copy()

sc.pp.neighbors(adata)
sc.pp.log1p(adata)


# In[7]:


### select top 3000 hvg for analysis #####
sc.pp.highly_variable_genes(
    adata,
    flavor="seurat_v3",
    n_top_genes=3_000,
    subset=True
)


# ## here we read the tf and targets from grn boost

# In[8]:


######### read the adj file and look for TF targets
######### select targest above 0.99 qauntile of importance 
adjacencies = pd.read_csv("adj.csv", index_col=False)
value = np.quantile(adjacencies['importance'].copy(), q = 0.98)
adj2 = adjacencies.copy()
adj2 = adjacencies[adjacencies['importance']>= value]
adj2


# ## here we select the tf and targets with importance above 0.98 quantile
# . rearrange the tf and targets list

# In[9]:


adj2.groupby('TF')['target'].agg(', '.join).reset_index().head(1000)
adj2.query("TF == 'OTX2'")


# In[10]:


adj3 = adj2.groupby('TF')['target'].agg(', '.join).reset_index().head(1000)
adj3


# In[10]:


adj3.to_csv('adj3.csv')


# ## time for cell oracle

# In[11]:


# Load base grn
base_GRN = co.data.load_human_promoter_base_GRN()

# Check data
base_GRN.head()


# In[12]:


# Make dictionary: dictionary key is TF and dictionary value is list of target genes.
TF_to_TG_dictionary = {}

for TF, TGs in zip(adj3.TF, adj3.target):
    # convert target gene to list
    TG_list = TGs.replace(" ", "").split(",")
    # store target gene list in a dictionary
    TF_to_TG_dictionary[TF] = TG_list

# We invert the dictionary above using a utility function in celloracle.
TG_to_TF_dictionary = co.utility.inverse_dictionary(TF_to_TG_dictionary)


# In[13]:


# we make a copy of adata for further analysis 
adata2 = adata.copy()


# In[14]:


# Show data name in anndata
print("metadata columns :", list(adata2.obs.columns))
print("dimensional reduction: ", list(adata2.obsm.keys()))


# In[15]:


# In this notebook, we use the unscaled mRNA count for the nput of Oracle object.
# N.B it will gibe a warning WARNING: adata.X seems to be already log-transformed. because there is decimal but the matrix is raw
adata2.X = adata2.layers["counts"].copy()

# Instantiate Oracle object
## i will name it with the gene i would like to purturb 
oracle = co.Oracle()
oracle.import_anndata_as_raw_count(adata=adata,
                                   cluster_column_name="louvain",
                                   embedding_name="X_draw_graph_fa")


# In[16]:


# cross confirm
adata.layers["counts"][:10,:10].todense()


# In[17]:


# You can load TF info dataframe with the following code.
oracle.import_TF_data(TF_info_matrix=base_GRN)


# In[18]:


oracle.addTFinfo_dictionary(TG_to_TF_dictionary)


# In[19]:


# Perform PCA
oracle.perform_PCA()

# Select important PCs
plt.plot(np.cumsum(oracle.pca.explained_variance_ratio_)[:100])
n_comps = np.where(np.diff(np.diff(np.cumsum(oracle.pca.explained_variance_ratio_))>0.002))[0][0]
plt.axvline(n_comps, c="k")
print(n_comps)
n_comps = min(n_comps, 50)


# In[20]:


n_cell = oracle.adata.shape[0]
print(f"cell number is :{n_cell}")


# In[21]:


k = int(0.025*n_cell)
print(f"Auto-selected k is :{k}")


# In[22]:


oracle.knn_imputation(n_pca_dims=n_comps, k=k, balanced=True, b_sight=k*8,
                      b_maxl=k*4, n_jobs=4)


# ## time for GRN Calculation

# In[23]:


sc.pl.draw_graph(oracle.adata, color=['louvain',"tissue"], ncols = 3,wspace= 0.5, frameon= False)


# In[24]:


get_ipython().run_cell_magic('time', '', '# Calculate GRN for each louvain clusters in MHB\n\nlinks = oracle.get_links(cluster_name_for_GRN_unit="louvain", alpha=10,\n                         verbose_level=10)\n')


# In[25]:


links.links_dict.keys()


# In[26]:


links.links_dict['1']


# In[27]:


links.palette


# ## filter uncertain networks and keep top 2000 edges

# In[28]:


links.filter_links(p=0.001, weight="coef_abs", threshold_number=100)
links.plot_degree_distributions(plot_model=True)


# ## time to calculate network scores

# In[29]:


links.get_network_score()


# In[30]:


links.merged_score.head(20)


# ## Network analysis; Network score for each gene

# In[31]:


# Check cluster name
links.cluster


# In[32]:


plt.rcParams["figure.figsize"] = [4, 6]


# In[33]:


# Visualize top n-th genes with high scores.
links.plot_scores_as_rank(cluster="4", n_gene=20)


# In[34]:


cluster_name = "0"
filtered_links_df = links.filtered_links[cluster_name]
filtered_links_df.head()


# In[35]:


### save the oracle object and links#######
oracle.to_hdf5("mistr_mhb.celloracle.oracle")
links.to_hdf5(file_path="mistr_mhb.celloracle.links")


# ## in silico perturbation

# In[36]:


oracle


# ## Make predictive models for simulation

# In[37]:


links.filter_links()
oracle.get_cluster_specific_TFdict_from_Links(links_object=links)
oracle.fit_GRN_for_simulation(alpha=10, use_cluster_specific_TFdict=True)


# In[38]:


plt.rcParams["figure.figsize"] = [4, 4]


# In[ ]:





# # simulate KO of PAX2

# In[283]:


# Check gene expression
goi = "PAX2"
sc.pl.draw_graph(oracle.adata, color=[goi, oracle.cluster_column_name],
                 layer="imputed_count", use_raw=False, cmap="viridis")


# In[284]:


# Plot gene expression in histogram
sc.get.obs_df(oracle.adata, keys=[goi], layer="imputed_count").hist()
plt.show()


# In[285]:


# i will create a copy of oracle objext to gene i want to knock out (PAX2)
oracle_PAX2 = oracle.copy()


# In[286]:


oracle_PAX2.simulate_shift(perturb_condition={goi: 0.0},
                      n_propagation=3)


# In[287]:


# Get transition probability
oracle_PAX2.estimate_transition_prob(n_neighbors=200,
                                knn_random=True,
                                sampled_fraction=1)

# Calculate embedding
oracle_PAX2.calculate_embedding_shift(sigma_corr=0.05)


# ## visualise KO simulation
# 

# In[288]:


fig, ax = plt.subplots(1, 2,  figsize=[13, 6])

scale = 20
# Show quiver plot
oracle_PAX2.plot_quiver(scale=scale, ax=ax[0])
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

# Show quiver plot that was calculated with randomized graph.
oracle_PAX2.plot_quiver_random(scale=scale, ax=ax[1])
ax[1].set_title(f"Randomized simulation vector")

plt.show()


# ## find paprameters for n grid and min mass
# 

# In[289]:


# n_grid = 40 is a good starting value.
n_grid = 40
oracle_PAX2.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)


# In[290]:


# Search for best min_mass.
oracle_PAX2.suggest_mass_thresholds(n_suggestion=12)


# In[291]:


min_mass = 0.0043
oracle_PAX2.calculate_mass_filter(min_mass=min_mass, plot=True)


# In[292]:


fig, ax = plt.subplots(1, 2,  figsize=[13, 6])

scale_simulation = 0.4
# Show quiver plot
oracle_PAX2.plot_simulation_flow_on_grid(scale=scale_simulation, ax=ax[0])
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

# Show quiver plot that was calculated with randomized graph.
oracle_PAX2.plot_simulation_flow_random_on_grid(scale=scale_simulation, ax=ax[1])
ax[1].set_title(f"Randomized simulation vector")

plt.show()


# In[293]:


# Plot vector field with cell cluster
fig, ax = plt.subplots(figsize=[7, 7])

oracle_PAX2.plot_cluster_whole(ax=ax, s=10)
oracle_PAX2.plot_simulation_flow_on_grid(scale=scale_simulation, ax=ax, show_background=False)
plt.savefig("co_plots/PAX2KO.png", dpi=150, bbox_inches = "tight")


# In[294]:


#######plot and save the vectors without the cluster colors 
fig, ax = plt.subplots(figsize=[7, 7])
oracle_PAX2.plot_simulation_flow_on_grid(scale=scale_simulation, ax=ax)
ax.set_title(f"Simulated cell identity shift vector: {goi} KO")

plt.savefig("co_plots/PAX2KO_wo_clusters.png", dpi=150, bbox_inches = "tight")
plt.show()


# ## simulate KO of GBX2

# In[295]:


# Check gene expression
goi = "GBX2"
sc.pl.draw_graph(oracle.adata, color=[goi, oracle.cluster_column_name],
                 layer="imputed_count", use_raw=False, cmap="viridis")


# In[296]:


# Plot gene expression in histogram
sc.get.obs_df(oracle.adata, keys=[goi], layer="imputed_count").hist()
plt.show()


# In[297]:


# i will create a copy of oracle objext to gene i want to knock out (GBX2)
oracle_GBX2 = oracle.copy()


# In[298]:


oracle_GBX2.simulate_shift(perturb_condition={goi: 0.0},
                      n_propagation=3)


# In[299]:


# Get transition probability
oracle_GBX2.estimate_transition_prob(n_neighbors=200,
                                knn_random=True,
                                sampled_fraction=1)

# Calculate embedding
oracle_GBX2.calculate_embedding_shift(sigma_corr=0.05)


# ## visualise KO simulation

# In[300]:


fig, ax = plt.subplots(1, 2,  figsize=[13, 6])

scale = 5
# Show quiver plot
oracle_GBX2.plot_quiver(scale=scale, ax=ax[0])
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

# Show quiver plot that was calculated with randomized graph.
oracle_GBX2.plot_quiver_random(scale=scale, ax=ax[1])
ax[1].set_title(f"Randomized simulation vector")

plt.show()


# ## find paprameters for n grid and min mass

# In[301]:


# n_grid = 40 is a good starting value.
n_grid = 40
oracle_GBX2.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)


# In[302]:


# Search for best min_mass.
oracle_GBX2.suggest_mass_thresholds(n_suggestion=12)


# In[303]:


min_mass = 0.0043
oracle_GBX2.calculate_mass_filter(min_mass=min_mass, plot=True)


# In[304]:


# Plot vector field with cell cluster
fig, ax = plt.subplots(figsize=[7, 7])

oracle_GBX2.plot_cluster_whole(ax=ax, s=10)
oracle_GBX2.plot_simulation_flow_on_grid(scale=scale_simulation, ax=ax, show_background=False)
plt.savefig("co_plots/GBX2KO.png", dpi=150, bbox_inches = "tight")


# In[305]:


#######plot and save the vectors without the cluster colors 
fig, ax = plt.subplots(figsize=[7, 7])
oracle_GBX2.plot_simulation_flow_on_grid(scale=scale_simulation, ax=ax)
ax.set_title(f"Simulated cell identity shift vector: {goi} KO")

plt.savefig("co_plots/GBX2KO_wo_clusters.png", dpi=150, bbox_inches = "tight")
plt.show()


# ## simulate KO of OTX2

# In[306]:


# Check gene expression
goi = "OTX2"
sc.pl.draw_graph(oracle.adata, color=[goi, oracle.cluster_column_name],
                 layer="imputed_count", use_raw=False, cmap="viridis")


# In[307]:


# Plot gene expression in histogram
sc.get.obs_df(oracle.adata, keys=[goi], layer="imputed_count").hist()
plt.show()


# In[308]:


# i will create a copy of oracle objext to gene i want to knock out (OTX2)
oracle_OTX2 = oracle.copy()


# In[309]:


oracle_OTX2.simulate_shift(perturb_condition={goi: 0.0},
                      n_propagation=3)


# In[310]:


# Get transition probability
oracle_OTX2.estimate_transition_prob(n_neighbors=200,
                                knn_random=True,
                                sampled_fraction=1)

# Calculate embedding
oracle_OTX2.calculate_embedding_shift(sigma_corr=0.05)


# ## visualise KO simulation

# In[311]:


fig, ax = plt.subplots(1, 2,  figsize=[13, 6])

scale = 25
# Show quiver plot
oracle_OTX2.plot_quiver(scale=scale, ax=ax[0])
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

# Show quiver plot that was calculated with randomized graph.
oracle_OTX2.plot_quiver_random(scale=scale, ax=ax[1])
ax[1].set_title(f"Randomized simulation vector")

plt.show()


# In[312]:


## find paprameters for n grid and min mass


# In[313]:


# n_grid = 40 is a good starting value.
n_grid = 40
oracle_OTX2.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)


# In[314]:


# Search for best min_mass.
oracle_OTX2.suggest_mass_thresholds(n_suggestion=12)


# In[315]:


min_mass = 0.0043
oracle_OTX2.calculate_mass_filter(min_mass=min_mass, plot=True)


# In[316]:


# Plot vector field with cell cluster
fig, ax = plt.subplots(figsize=[7, 7])

oracle_OTX2.plot_cluster_whole(ax=ax, s=10)
oracle_OTX2.plot_simulation_flow_on_grid(scale=scale_simulation, ax=ax, show_background=False)
plt.savefig("co_plots/OTX2KO.png", dpi=150, bbox_inches = "tight")


# In[317]:


#######plot and save the vectors without the cluster colors 
fig, ax = plt.subplots(figsize=[7, 7])
oracle_OTX2.plot_simulation_flow_on_grid(scale=scale_simulation, ax=ax)
ax.set_title(f"Simulated cell identity shift vector: {goi} KO")

plt.savefig("co_plots/OTX2KO_wo_clusters.png", dpi=150, bbox_inches = "tight")
plt.show()


# ## simulate KO of EN1

# In[318]:


# Check gene expression
goi = "EN1"
sc.pl.draw_graph(oracle.adata, color=[goi, oracle.cluster_column_name],
                 layer="imputed_count", use_raw=False, cmap="viridis")


# In[319]:


# Plot gene expression in histogram
sc.get.obs_df(oracle.adata, keys=[goi], layer="imputed_count").hist()
plt.show()


# In[320]:


# i will create a copy of oracle objext to gene i want to knock out (EN1)
oracle_EN1 = oracle.copy()


# In[321]:


oracle_EN1.simulate_shift(perturb_condition={goi: 0.0},
                      n_propagation=3)


# In[322]:


# Get transition probability
oracle_EN1.estimate_transition_prob(n_neighbors=200,
                                knn_random=True,
                                sampled_fraction=1)

# Calculate embedding
oracle_EN1.calculate_embedding_shift(sigma_corr=0.05)


# ## visualise KO simulation

# In[323]:


fig, ax = plt.subplots(1, 2,  figsize=[13, 6])

scale = 25
# Show quiver plot
oracle_EN1.plot_quiver(scale=scale, ax=ax[0])
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

# Show quiver plot that was calculated with randomized graph.
oracle_EN1.plot_quiver_random(scale=scale, ax=ax[1])
ax[1].set_title(f"Randomized simulation vector")

plt.show()


# In[324]:


## find paprameters for n grid and min mass


# In[325]:


# n_grid = 40 is a good starting value.
n_grid = 40
oracle_EN1.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)


# In[326]:


# Search for best min_mass.
oracle_EN1.suggest_mass_thresholds(n_suggestion=12)


# In[327]:


min_mass = 0.0043
oracle_EN1.calculate_mass_filter(min_mass=min_mass, plot=True)


# In[328]:


# Plot vector field with cell cluster
fig, ax = plt.subplots(figsize=[7, 7])

oracle_EN1.plot_cluster_whole(ax=ax, s=10)
oracle_EN1.plot_simulation_flow_on_grid(scale=scale_simulation, ax=ax, show_background=False)
plt.savefig("co_plots/EN1KO.png", dpi=150, bbox_inches = "tight")


# In[329]:


#######plot and save the vectors without the cluster colors 
fig, ax = plt.subplots(figsize=[7, 7])
oracle_EN1.plot_simulation_flow_on_grid(scale=scale_simulation, ax=ax)
ax.set_title(f"Simulated cell identity shift vector: {goi} KO")

plt.savefig("co_plots/EN1KO_wo_clusters.png", dpi=150, bbox_inches = "tight")
plt.show()


# ## simulate KO of DMBX1

# In[330]:


# Check gene expression
goi = "DMBX1"
sc.pl.draw_graph(oracle.adata, color=[goi, oracle.cluster_column_name],
                 layer="imputed_count", use_raw=False, cmap="viridis")


# In[331]:


# Plot gene expression in histogram
sc.get.obs_df(oracle.adata, keys=[goi], layer="imputed_count").hist()
plt.show()


# In[332]:


# i will create a copy of oracle objext to gene i want to knock out (DMBX1)
oracle_DMBX1 = oracle.copy()


# In[333]:


oracle_DMBX1.simulate_shift(perturb_condition={goi: 0.0},
                      n_propagation=3)


# In[334]:


# Get transition probability
oracle_DMBX1.estimate_transition_prob(n_neighbors=200,
                                knn_random=True,
                                sampled_fraction=1)

# Calculate embedding
oracle_DMBX1.calculate_embedding_shift(sigma_corr=0.05)


# ## visualise KO simulation

# In[335]:


fig, ax = plt.subplots(1, 2,  figsize=[13, 6])

scale = 5
# Show quiver plot
oracle_DMBX1.plot_quiver(scale=scale, ax=ax[0])
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

# Show quiver plot that was calculated with randomized graph.
oracle_DMBX1.plot_quiver_random(scale=scale, ax=ax[1])
ax[1].set_title(f"Randomized simulation vector")

plt.show()


# In[336]:


## find parameters for n grid and min mass


# In[337]:


# n_grid = 40 is a good starting value.
n_grid = 40
oracle_DMBX1.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)


# In[338]:


# Search for best min_mass.
oracle_DMBX1.suggest_mass_thresholds(n_suggestion=12)


# In[339]:


min_mass = 0.0043
oracle_DMBX1.calculate_mass_filter(min_mass=min_mass, plot=True)


# In[340]:


# Plot vector field with cell cluster
fig, ax = plt.subplots(figsize=[7, 7])

oracle_DMBX1.plot_cluster_whole(ax=ax, s=10)
oracle_DMBX1.plot_simulation_flow_on_grid(scale=scale_simulation, ax=ax, show_background=False)
plt.savefig("co_plots/DMBX1KO.png", dpi=150, bbox_inches = "tight")


# In[341]:


#######plot and save the vectors without the cluster colors 
fig, ax = plt.subplots(figsize=[7, 7])
oracle_DMBX1.plot_simulation_flow_on_grid(scale=scale_simulation, ax=ax)
ax.set_title(f"Simulated cell identity shift vector: {goi} KO")

plt.savefig("co_plots/DMBX1KO_wo_clusters.png", dpi=150, bbox_inches = "tight")
plt.show()


# ## simulate KO of PAX5

# In[342]:


# Check gene expression
goi = "PAX5"
sc.pl.draw_graph(oracle.adata, color=[goi, oracle.cluster_column_name],
                 layer="imputed_count", use_raw=False, cmap="viridis")


# In[343]:


# Plot gene expression in histogram
sc.get.obs_df(oracle.adata, keys=[goi], layer="imputed_count").hist()
plt.show()


# In[344]:


# i will create a copy of oracle objext to gene i want to knock out (PAX5)
oracle_PAX5 = oracle.copy()


# In[345]:


oracle_PAX5.simulate_shift(perturb_condition={goi: 0.0},
                      n_propagation=3)


# In[346]:


# Get transition probability
oracle_PAX5.estimate_transition_prob(n_neighbors=200,
                                knn_random=True,
                                sampled_fraction=1)

# Calculate embedding
oracle_PAX5.calculate_embedding_shift(sigma_corr=0.05)


# ## visualise KO simulation

# In[347]:


fig, ax = plt.subplots(1, 2,  figsize=[13, 6])

scale = 20
# Show quiver plot
oracle_PAX5.plot_quiver(scale=scale, ax=ax[0])
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

# Show quiver plot that was calculated with randomized graph.
oracle_PAX5.plot_quiver_random(scale=scale, ax=ax[1])
ax[1].set_title(f"Randomized simulation vector")

plt.show()


# In[348]:


## find parameters for n grid and min mass


# In[349]:


## find parameters for n grid and min mass
n_grid = 40
oracle_PAX5.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)


# In[350]:


# Search for best min_mass.
oracle_PAX5.suggest_mass_thresholds(n_suggestion=12)


# In[351]:


min_mass = 0.0043
oracle_PAX5.calculate_mass_filter(min_mass=min_mass, plot=True)


# In[352]:


# Plot vector field with cell cluster
fig, ax = plt.subplots(figsize=[7, 7])

oracle_PAX5.plot_cluster_whole(ax=ax, s=10)
oracle_PAX5.plot_simulation_flow_on_grid(scale=scale_simulation, ax=ax, show_background=False)
plt.savefig("co_plots/PAX5.png", dpi=150, bbox_inches = "tight")


# In[353]:


#######plot and save the vectors without the cluster colors 
fig, ax = plt.subplots(figsize=[7, 7])
oracle_PAX5.plot_simulation_flow_on_grid(scale=scale_simulation, ax=ax)
ax.set_title(f"Simulated cell identity shift vector: {goi} KO")

plt.savefig("co_plots/PAX5KO_wo_clusters.png", dpi=150, bbox_inches = "tight")
plt.show()


# ## simulate KO of PAX8

# In[354]:


# Check gene expression
goi = "PAX8"
sc.pl.draw_graph(oracle.adata, color=[goi, oracle.cluster_column_name],
                 layer="imputed_count", use_raw=False, cmap="viridis")


# In[355]:


# Plot gene expression in histogram
sc.get.obs_df(oracle.adata, keys=[goi], layer="imputed_count").hist()
plt.show()


# In[356]:


# i will create a copy of oracle objext to gene i want to knock out (PAX8)
oracle_PAX8 = oracle.copy()


# In[357]:


oracle_PAX8.simulate_shift(perturb_condition={goi: 0.0},
                      n_propagation=3)


# In[358]:


# Get transition probability
oracle_PAX8.estimate_transition_prob(n_neighbors=200,
                                knn_random=True,
                                sampled_fraction=1)

# Calculate embedding
oracle_PAX8.calculate_embedding_shift(sigma_corr=0.05)


# ## visualise KO simulation 

# In[359]:


fig, ax = plt.subplots(1, 2,  figsize=[13, 6])

scale = 25
# Show quiver plot
oracle_PAX8.plot_quiver(scale=scale, ax=ax[0])
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

# Show quiver plot that was calculated with randomized graph.
oracle_PAX8.plot_quiver_random(scale=scale, ax=ax[1])
ax[1].set_title(f"Randomized simulation vector")

plt.show()


# In[360]:


## find parameters for n grid and min mass
n_grid = 40
oracle_PAX8.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)


# In[361]:


# Search for best min_mass.
oracle_PAX8.suggest_mass_thresholds(n_suggestion=12)


# In[362]:


min_mass = 0.0043
oracle_PAX8.calculate_mass_filter(min_mass=min_mass, plot=True)


# In[363]:


# Plot vector field with cell cluster
fig, ax = plt.subplots(figsize=[7, 7])

oracle_PAX8.plot_cluster_whole(ax=ax, s=10)
oracle_PAX8.plot_simulation_flow_on_grid(scale=scale_simulation, ax=ax, show_background=False)
plt.savefig("co_plots/PAX8.png", dpi=150, bbox_inches = "tight")


# In[364]:


#######plot and save the vectors without the cluster colors 
fig, ax = plt.subplots(figsize=[7, 7])
oracle_PAX8.plot_simulation_flow_on_grid(scale=scale_simulation, ax=ax)
ax.set_title(f"Simulated cell identity shift vector: {goi} KO")

plt.savefig("co_plots/PAX8KO_wo_clusters.png", dpi=150, bbox_inches = "tight")
plt.show()


# ## simulate KO of EN2

# In[56]:


# Check gene expression
goi = "EN2"
sc.pl.draw_graph(oracle.adata, color=[goi, oracle.cluster_column_name],
                 layer="imputed_count", use_raw=False, cmap="viridis")


# In[57]:


# Plot gene expression in histogram
sc.get.obs_df(oracle.adata, keys=[goi], layer="imputed_count").hist()
plt.show()


# In[58]:


# i will create a copy of oracle objext to gene i want to knock out (EN2)
oracle_EN2 = oracle.copy()


# In[59]:


oracle_EN2.simulate_shift(perturb_condition={goi: 0.0},
                      n_propagation=3)


# In[60]:


# Get transition probability
oracle_EN2.estimate_transition_prob(n_neighbors=200,
                                knn_random=True,
                                sampled_fraction=1)

# Calculate embedding
oracle_EN2.calculate_embedding_shift(sigma_corr=0.05)


# ## visualise KO simulation

# In[61]:


fig, ax = plt.subplots(1, 2,  figsize=[13, 6])

scale = 20
# Show quiver plot
oracle_EN2.plot_quiver(scale=scale, ax=ax[0])
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

# Show quiver plot that was calculated with randomized graph.
oracle_EN2.plot_quiver_random(scale=scale, ax=ax[1])
ax[1].set_title(f"Randomized simulation vector")

plt.show()


# In[62]:


## find parameters for n grid and min mass
n_grid = 40
oracle_EN2.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)


# In[63]:


# Search for best min_mass.
oracle_EN2.suggest_mass_thresholds(n_suggestion=12)


# In[64]:


min_mass = 0.0043
oracle_EN2.calculate_mass_filter(min_mass=min_mass, plot=True)


# In[65]:


# Plot vector field with cell cluster
fig, ax = plt.subplots(figsize=[7, 7])

oracle_EN2.plot_cluster_whole(ax=ax, s=10)
oracle_EN2.plot_simulation_flow_on_grid(scale=0.2, ax=ax, show_background=False)
plt.savefig("co_plots/EN2.png", dpi=150, bbox_inches = "tight")


# In[66]:


#######plot and save the vectors without the cluster colors 
fig, ax = plt.subplots(figsize=[7, 7])
oracle_EN2.plot_simulation_flow_on_grid(scale=0.2, ax=ax)
ax.set_title(f"Simulated cell identity shift vector: {goi} KO")

plt.savefig("co_plots/EN2KO_wo_clusters.png", dpi=150, bbox_inches = "tight")
plt.show()


# ## simulate KO of MSX1

# In[73]:


# Check gene expression
goi = "MSX1"
sc.pl.draw_graph(oracle.adata, color=[goi, oracle.cluster_column_name],
                 layer="imputed_count", use_raw=False, cmap="viridis")


# In[74]:


# Plot gene expression in histogram
sc.get.obs_df(oracle.adata, keys=[goi], layer="imputed_count").hist()
plt.show()


# In[75]:


# i will create a copy of oracle objext to gene i want to knock out (EN2)
oracle_MSX1 = oracle.copy()


# In[76]:


oracle_MSX1.simulate_shift(perturb_condition={goi: 0.0},
                      n_propagation=3)


# In[77]:


# Get transition probability
oracle_MSX1.estimate_transition_prob(n_neighbors=200,
                                knn_random=True,
                                sampled_fraction=1)

# Calculate embedding
oracle_MSX1.calculate_embedding_shift(sigma_corr=0.05)


# ## simulate KO simulation

# In[78]:


fig, ax = plt.subplots(1, 2,  figsize=[13, 6])

scale = 20
# Show quiver plot
oracle_MSX1.plot_quiver(scale=scale, ax=ax[0])
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

# Show quiver plot that was calculated with randomized graph.
oracle_MSX1.plot_quiver_random(scale=scale, ax=ax[1])
ax[1].set_title(f"Randomized simulation vector")

plt.show()


# In[79]:


## find parameters for n grid and min mass
n_grid = 40
oracle_MSX1.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)


# In[80]:


min_mass = 0.0043
oracle_MSX1.calculate_mass_filter(min_mass=min_mass, plot=True)


# In[81]:


# Plot vector field with cell cluster
fig, ax = plt.subplots(figsize=[7, 7])

oracle_MSX1.plot_cluster_whole(ax=ax, s=10)
oracle_MSX1.plot_simulation_flow_on_grid(scale=0.2, ax=ax, show_background=False)
plt.savefig("co_plots/MSX1.png", dpi=150, bbox_inches = "tight")


# In[82]:


#######plot and save the vectors without the cluster colors 
fig, ax = plt.subplots(figsize=[7, 7])
oracle_MSX1.plot_simulation_flow_on_grid(scale=0.2, ax=ax)
ax.set_title(f"Simulated cell identity shift vector: {goi} KO")

plt.savefig("co_plots/MSX1KO_wo_clusters.png", dpi=150, bbox_inches = "tight")
plt.show()


# ## simulate KO of MSX2

# In[83]:


# Check gene expression
goi = "MSX2"
sc.pl.draw_graph(oracle.adata, color=[goi, oracle.cluster_column_name],
                 layer="imputed_count", use_raw=False, cmap="viridis")


# In[84]:


# Plot gene expression in histogram
sc.get.obs_df(oracle.adata, keys=[goi], layer="imputed_count").hist()
plt.show()


# In[85]:


# i will create a copy of oracle objext to gene i want to knock out (EN2)
oracle_MSX2 = oracle.copy()


# In[86]:


oracle_MSX2.simulate_shift(perturb_condition={goi: 0.0},
                      n_propagation=3)


# In[87]:


# Get transition probability
oracle_MSX2.estimate_transition_prob(n_neighbors=200,
                                knn_random=True,
                                sampled_fraction=1)

# Calculate embedding
oracle_MSX2.calculate_embedding_shift(sigma_corr=0.05)


# ## simulate KO simulation

# In[89]:


fig, ax = plt.subplots(1, 2,  figsize=[13, 6])

scale = 10
# Show quiver plot
oracle_MSX2.plot_quiver(scale=scale, ax=ax[0])
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

# Show quiver plot that was calculated with randomized graph.
oracle_MSX2.plot_quiver_random(scale=scale, ax=ax[1])
ax[1].set_title(f"Randomized simulation vector")

plt.show()


# In[90]:


## find parameters for n grid and min mass
n_grid = 40
oracle_MSX2.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)


# In[91]:


min_mass = 0.0043
oracle_MSX2.calculate_mass_filter(min_mass=min_mass, plot=True)


# In[92]:


# Plot vector field with cell cluster
fig, ax = plt.subplots(figsize=[7, 7])

oracle_MSX2.plot_cluster_whole(ax=ax, s=10)
oracle_MSX2.plot_simulation_flow_on_grid(scale=0.2, ax=ax, show_background=False)
plt.savefig("co_plots/MSX2.png", dpi=150, bbox_inches = "tight")


# In[93]:


#######plot and save the vectors without the cluster colors 
fig, ax = plt.subplots(figsize=[7, 7])
oracle_MSX2.plot_simulation_flow_on_grid(scale=0.2, ax=ax)
ax.set_title(f"Simulated cell identity shift vector: {goi} KO")

plt.savefig("co_plots/MSX2KO_wo_clusters.png", dpi=150, bbox_inches = "tight")
plt.show()


# In[ ]:





# ## simulate KO of SP9

# In[94]:


# Check gene expression
goi = "SP9"
sc.pl.draw_graph(oracle.adata, color=[goi, oracle.cluster_column_name],
                 layer="imputed_count", use_raw=False, cmap="viridis")


# In[95]:


# Plot gene expression in histogram
sc.get.obs_df(oracle.adata, keys=[goi], layer="imputed_count").hist()
plt.show()


# In[96]:


# i will create a copy of oracle objext to gene i want to knock out (SP9)
oracle_SP9 = oracle.copy()


# In[98]:


oracle_SP9.simulate_shift(perturb_condition={goi: 0.0},
                      n_propagation=3)


# In[99]:


# Get transition probability
oracle_SP9.estimate_transition_prob(n_neighbors=200,
                                knn_random=True,
                                sampled_fraction=1)

# Calculate embedding
oracle_SP9.calculate_embedding_shift(sigma_corr=0.05)


# In[100]:


## simulate KO simulation


# In[102]:


fig, ax = plt.subplots(1, 2,  figsize=[13, 6])

scale = 10
# Show quiver plot
oracle_SP9.plot_quiver(scale=scale, ax=ax[0])
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

# Show quiver plot that was calculated with randomized graph.
oracle_SP9.plot_quiver_random(scale=scale, ax=ax[1])
ax[1].set_title(f"Randomized simulation vector")

plt.show()


# In[104]:


## find parameters for n grid and min mass
n_grid = 40
oracle_SP9.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)


# In[105]:


min_mass = 0.0043
oracle_SP9.calculate_mass_filter(min_mass=min_mass, plot=True)


# In[109]:


# Plot vector field with cell cluster
fig, ax = plt.subplots(figsize=[7, 7])

oracle_SP9.plot_cluster_whole(ax=ax, s=10)
oracle_SP9.plot_simulation_flow_on_grid(scale=0.4, ax=ax, show_background=False)
plt.savefig("co_plots/SP9.png", dpi=150, bbox_inches = "tight")


# In[110]:


#######plot and save the vectors without the cluster colors 
fig, ax = plt.subplots(figsize=[7, 7])
oracle_SP9.plot_simulation_flow_on_grid(scale=0.4, ax=ax)
ax.set_title(f"Simulated cell identity shift vector: {goi} KO")

plt.savefig("co_plots/SP9KO_wo_clusters.png", dpi=150, bbox_inches = "tight")
plt.show()


# ## simulate KO of FOXJ1

# In[111]:


# Check gene expression
goi = "FOXJ1"
sc.pl.draw_graph(oracle.adata, color=[goi, oracle.cluster_column_name],
                 layer="imputed_count", use_raw=False, cmap="viridis")


# In[112]:


# Plot gene expression in histogram
sc.get.obs_df(oracle.adata, keys=[goi], layer="imputed_count").hist()
plt.show()


# In[113]:


# i will create a copy of oracle objext to gene i want to knock out (EN2)
oracle_FOXJ1= oracle.copy()


# In[114]:


oracle_FOXJ1.simulate_shift(perturb_condition={goi: 0.0},
                      n_propagation=3)


# In[115]:


# Get transition probability
oracle_FOXJ1.estimate_transition_prob(n_neighbors=200,
                                knn_random=True,
                                sampled_fraction=1)

# Calculate embedding
oracle_FOXJ1.calculate_embedding_shift(sigma_corr=0.05)


# ## simulate KO simulation

# In[118]:


fig, ax = plt.subplots(1, 2,  figsize=[13, 6])

scale = 10
# Show quiver plot
oracle_FOXJ1.plot_quiver(scale=scale, ax=ax[0])
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

# Show quiver plot that was calculated with randomized graph.
oracle_FOXJ1.plot_quiver_random(scale=scale, ax=ax[1])
ax[1].set_title(f"Randomized simulation vector")

plt.show()


# In[117]:


## find parameters for n grid and min mass
n_grid = 40
oracle_FOXJ1.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)


# In[119]:


min_mass = 0.0043
oracle_FOXJ1.calculate_mass_filter(min_mass=min_mass, plot=True)


# In[123]:


# Plot vector field with cell cluster
fig, ax = plt.subplots(figsize=[7, 7])

oracle_FOXJ1.plot_cluster_whole(ax=ax, s=10)
oracle_FOXJ1.plot_simulation_flow_on_grid(scale=0.4, ax=ax, show_background=False)
plt.savefig("co_plots/FOXJ1.png", dpi=150, bbox_inches = "tight")


# In[122]:


#######plot and save the vectors without the cluster colors 
fig, ax = plt.subplots(figsize=[7, 7])
oracle_FOXJ1.plot_simulation_flow_on_grid(scale=0.4, ax=ax)
ax.set_title(f"Simulated cell identity shift vector: {goi} KO")

plt.savefig("co_plots/FOXJ1KO_wo_clusters.png", dpi=150, bbox_inches = "tight")
plt.show()


# In[ ]:





# ## simulate KO of POU3F4

# In[125]:


# Check gene expression
goi = "POU3F4"
sc.pl.draw_graph(oracle.adata, color=[goi, oracle.cluster_column_name],
                 layer="imputed_count", use_raw=False, cmap="viridis")


# In[126]:


# Plot gene expression in histogram
sc.get.obs_df(oracle.adata, keys=[goi], layer="imputed_count").hist()
plt.show()


# In[127]:


# i will create a copy of oracle objext to gene i want to knock out (EN2)
oracle_POU3F4 = oracle.copy()


# In[128]:


oracle_POU3F4.simulate_shift(perturb_condition={goi: 0.0},
                      n_propagation=3)


# In[129]:


# Get transition probability
oracle_POU3F4.estimate_transition_prob(n_neighbors=200,
                                knn_random=True,
                                sampled_fraction=1)

# Calculate embedding
oracle_POU3F4.calculate_embedding_shift(sigma_corr=0.05)


# ## simulate KO simulation

# In[132]:


fig, ax = plt.subplots(1, 2,  figsize=[13, 6])

scale = 10
# Show quiver plot
oracle_POU3F4.plot_quiver(scale=scale, ax=ax[0])
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

# Show quiver plot that was calculated with randomized graph.
oracle_POU3F4.plot_quiver_random(scale=scale, ax=ax[1])
ax[1].set_title(f"Randomized simulation vector")

plt.show()


# In[131]:


## find parameters for n grid and min mass
n_grid = 40
oracle_POU3F4.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)


# In[133]:


min_mass = 0.0043
oracle_POU3F4.calculate_mass_filter(min_mass=min_mass, plot=True)


# In[134]:


# Plot vector field with cell cluster
fig, ax = plt.subplots(figsize=[7, 7])

oracle_POU3F4.plot_cluster_whole(ax=ax, s=10)
oracle_POU3F4.plot_simulation_flow_on_grid(scale=0.2, ax=ax, show_background=False)
plt.savefig("co_plots/POU3F4.png", dpi=150, bbox_inches = "tight")


# In[135]:


#######plot and save the vectors without the cluster colors 
fig, ax = plt.subplots(figsize=[7, 7])
oracle_POU3F4.plot_simulation_flow_on_grid(scale=0.2, ax=ax)
ax.set_title(f"Simulated cell identity shift vector: {goi} KO")

plt.savefig("co_plots/POU3F4KO_wo_clusters.png", dpi=150, bbox_inches = "tight")
plt.show()


# In[ ]:





# ## simulate KO of MAFB 

# In[152]:


# Check gene expression
goi = "MAFB"
sc.pl.draw_graph(oracle.adata, color=[goi, oracle.cluster_column_name],
                 layer="imputed_count", use_raw=False, cmap="viridis")


# In[153]:


# Plot gene expression in histogram
sc.get.obs_df(oracle.adata, keys=[goi], layer="imputed_count").hist()
plt.show()


# In[154]:


# i will create a copy of oracle objext to gene i want to knock out (EN2)
oracle_MAFB = oracle.copy()


# In[155]:


oracle_MAFB.simulate_shift(perturb_condition={goi: 0.0},
                      n_propagation=3)


# In[156]:


# Get transition probability
oracle_MAFB.estimate_transition_prob(n_neighbors=200,
                                knn_random=True,
                                sampled_fraction=1)

# Calculate embedding
oracle_MAFB.calculate_embedding_shift(sigma_corr=0.05)


# ## simulate KO simulation

# In[157]:


fig, ax = plt.subplots(1, 2,  figsize=[13, 6])

scale = 10
# Show quiver plot
oracle_MAFB.plot_quiver(scale=scale, ax=ax[0])
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

# Show quiver plot that was calculated with randomized graph.
oracle_MAFB.plot_quiver_random(scale=scale, ax=ax[1])
ax[1].set_title(f"Randomized simulation vector")

plt.show()


# In[158]:


## find parameters for n grid and min mass
n_grid = 40
oracle_MAFB.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)


# In[162]:


min_mass = 0.0043
oracle_MAFB.calculate_mass_filter(min_mass=min_mass, plot=True)


# In[163]:


# Plot vector field with cell cluster
fig, ax = plt.subplots(figsize=[7, 7])

oracle_MAFB.plot_cluster_whole(ax=ax, s=10)
oracle_MAFB.plot_simulation_flow_on_grid(scale=0.4, ax=ax, show_background=False)
plt.savefig("co_plots/MAFB.png", dpi=150, bbox_inches = "tight")


# In[164]:


#######plot and save the vectors without the cluster colors 
fig, ax = plt.subplots(figsize=[7, 7])
oracle_MAFB.plot_simulation_flow_on_grid(scale=0.2, ax=ax)
ax.set_title(f"Simulated cell identity shift vector: {goi} KO")

plt.savefig("co_plots/MAFBKO_wo_clusters.png", dpi=150, bbox_inches = "tight")
plt.show()


# ## simulate KO of MSX1

# In[165]:


# Check gene expression
goi = "MEIS1"
sc.pl.draw_graph(oracle.adata, color=[goi, oracle.cluster_column_name],
                 layer="imputed_count", use_raw=False, cmap="viridis")


# In[166]:


# Plot gene expression in histogram
sc.get.obs_df(oracle.adata, keys=[goi], layer="imputed_count").hist()
plt.show()


# In[167]:


# i will create a copy of oracle objext to gene i want to knock out (EN2)
oracle_MEIS1 = oracle.copy()


# In[168]:


oracle_MEIS1.simulate_shift(perturb_condition={goi: 0.0},
                      n_propagation=3)


# In[169]:


# Get transition probability
oracle_MEIS1.estimate_transition_prob(n_neighbors=200,
                                knn_random=True,
                                sampled_fraction=1)

# Calculate embedding
oracle_MEIS1.calculate_embedding_shift(sigma_corr=0.05)


# In[170]:


## simulate KO simulation


# In[172]:


fig, ax = plt.subplots(1, 2,  figsize=[13, 6])

scale = 10
# Show quiver plot
oracle_MEIS1.plot_quiver(scale=scale, ax=ax[0])
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

# Show quiver plot that was calculated with randomized graph.
oracle_MEIS1.plot_quiver_random(scale=scale, ax=ax[1])
ax[1].set_title(f"Randomized simulation vector")

plt.show()


# In[173]:


## find parameters for n grid and min mass
n_grid = 40
oracle_MEIS1.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)


# In[174]:


min_mass = 0.0043
oracle_MEIS1.calculate_mass_filter(min_mass=min_mass, plot=True)


# In[175]:


# Plot vector field with cell cluster
fig, ax = plt.subplots(figsize=[7, 7])

oracle_MEIS1.plot_cluster_whole(ax=ax, s=10)
oracle_MEIS1.plot_simulation_flow_on_grid(scale=0.2, ax=ax, show_background=False)
plt.savefig("co_plots/MEIS1.png", dpi=150, bbox_inches = "tight")


# In[177]:


#######plot and save the vectors without the cluster colors 
fig, ax = plt.subplots(figsize=[7, 7])
oracle_MEIS1.plot_simulation_flow_on_grid(scale=0.4, ax=ax)
ax.set_title(f"Simulated cell identity shift vector: {goi} KO")

plt.savefig("co_plots/MEIS1KO_wo_clusters.png", dpi=150, bbox_inches = "tight")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




