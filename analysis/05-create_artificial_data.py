
# coding: utf-8

# In[1]:


import networkx
import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats as st
from sklearn import preprocessing
import pickle
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("../data_raw/gene_summary.tsv", sep="\t")


# In[3]:


genes = data.id.values


# In[4]:


G = networkx.read_edgelist(
      "../data_raw/mouse_gene_network.tsv",
      delimiter="\t",
      data=(('weight', float),),
      nodetype=str)


# In[5]:


essential_genes = np.array(list(genes[:4]) + ["POLR2C", "POLR1B", "PSMC1", "PSMD4", "TH"])


# In[6]:


neighbors = []
for c in essential_genes:
     neighbors += networkx.neighbors(G, c)
neighbors = np.unique(neighbors)


# In[7]:


G = G.subgraph(np.sort(neighbors))


# In[8]:


np.random.seed(42)
nonessential_genes = np.random.choice(list(G.nodes), size=30, replace=False)
filter_genes = np.append(essential_genes, nonessential_genes)


# In[9]:


G_filtered = G.subgraph(np.sort(filter_genes))


# In[10]:


essential_gene_map = {e: e for i, e in enumerate(list(G_filtered.nodes)) if e in essential_genes}


# In[11]:


networkx.draw(G_filtered, node_size=20, labels=essential_gene_map);


# In[12]:


cliques = [len(c) for c in networkx.clique.find_cliques(G_filtered)]


# In[13]:


plt.hist(cliques);


# ## Data generation small

# In[14]:


gamma_tau = .25
gamma_tau_non_essential = .1


# In[15]:


np.random.seed(1)
n_essential = 1
n_nonessential = 1
n_genes = n_essential + n_nonessential

gamma_essential = sp.random.normal(-1, scale=gamma_tau, size=n_essential)
gamma_nonessential = sp.random.normal(0, scale=gamma_tau_non_essential, size=n_nonessential)
gamma = sp.append(gamma_essential, gamma_nonessential)


# In[16]:


gamma_essential


# In[17]:


gamma_nonessential


# In[18]:


n_conditions, n_sgrnas, n_replicates = 5, 5, 5


# In[19]:


genes = essential_genes[:2]


# In[20]:


conditions = ["C" + str(i) for i in range(n_conditions)]
sgrnas = ["S" + str(i) for i in range(n_sgrnas)]
replicates = ["R" + str(i) for i in range(n_replicates)]


# In[21]:


combinations = [(g, c, s, r)      for g in genes for c in conditions      for s in sgrnas for r in replicates]


# In[24]:


count_table = pd.DataFrame(
    combinations, columns=["genes", "conditions", "sgrnas", "replicates"])


# In[26]:


count_table.shape


# In[27]:


sgrna_ids = np.repeat(["S" + str(i)                        for i in range(n_conditions * n_sgrnas * n_genes)], 
                      n_replicates)
count_table.sgrnas = sgrna_ids
condition_ids = np.repeat(["C" + str(i)                            for i in range(n_genes * n_conditions)],
                          n_sgrnas * n_replicates)
count_table.conditions = condition_ids


# In[28]:


le = preprocessing.LabelEncoder()
for i in count_table.columns.values:
    count_table[i] = le.fit_transform(count_table[i])


# In[29]:


beta_tau = .25
l_tau = .25


# In[31]:


beta = st.norm.rvs(np.repeat(gamma, n_conditions), beta_tau)
l = st.norm.rvs(0, l_tau, size = n_conditions * n_genes * n_sgrnas)


# In[32]:


data_tau = .25
data = st.norm.rvs(
    l[count_table["sgrnas"]] + beta[count_table["conditions"]],
    data_tau)


# In[33]:


count_table = pd.DataFrame(
    combinations, 
    columns=["gene", "condition", "intervention", "replicate"])

count_table["gamma"] = np.repeat(gamma, count_table.shape[0] / len(gamma))
count_table["beta"] = np.repeat(beta, count_table.shape[0] / len(beta))
count_table["l"] = np.repeat(l, count_table.shape[0] / len(l))
count_table["readout"] = data

sgrna_ids = np.repeat(["S" + str(i) for i in range(n_conditions * n_sgrnas * n_genes)], 
                      n_replicates)
count_table.intervention = sgrna_ids


# In[34]:


count_table


# In[35]:


count_table.to_csv("../data_raw/easy_simulated_data/small-simulated_data.tsv", index=False, sep="\t")


# In[36]:


G_filtered_small = G_filtered.subgraph(genes)


# In[37]:


networkx.readwrite.edgelist.write_weighted_edgelist(
    G_filtered_small, "../data_raw/easy_simulated_data/small-graph.tsv", delimiter="\t")


# In[39]:


data = {    
    "graph": G_filtered_small,
    "essential_genes": essential_genes,
    "nonessential_genes": nonessential_genes,    
    "gamma_tau": gamma_tau,
    "gamma_tau_non_essential": gamma_tau_non_essential,
    "gamma_essential": gamma_essential,
    "gamma_nonessential": gamma_nonessential,
    "gamma": gamma,
    "beta_tau": beta_tau,    
    "beta": beta,
    "l_tau": l_tau,
    "l": l,
    "data_tau": data_tau,
    "data": data,
    "count_table": count_table
}


# In[40]:


with open("../data_raw/easy_simulated_data/small-data.pickle","wb") as out:
    pickle.dump(data, out)


# ## Data generation big

# In[35]:


gamma_tau = .25
gamma_tau_non_essential = .1


# In[36]:


np.random.seed(1)
n_essential = len(essential_genes)
n_nonessential = len(nonessential_genes)
n_genes = n_essential + n_nonessential

gamma_essential = sp.random.normal(-1, scale=gamma_tau, size=n_essential)
gamma_nonessential = sp.random.normal(0, scale=gamma_tau_non_essential, size=n_nonessential)
gamma = sp.append(gamma_essential, gamma_nonessential)


# In[37]:


gamma_essential


# In[38]:


gamma_nonessential


# In[39]:


sp.log2(
    st.norm.pdf(gamma_essential, -1, gamma_tau) /  \
    st.norm.pdf(gamma_essential, 0, gamma_tau_non_essential)
)


# In[40]:


sp.log2(
    st.norm.pdf(gamma_nonessential, -1, gamma_tau) /  \
    st.norm.pdf(gamma_nonessential, 0, gamma_tau_non_essential)
)


# In[41]:


n_conditions, n_sgrnas, n_replicates = 4, 5, 5


# In[42]:


genes = filter_genes
conditions = ["C" + str(i) for i in range(n_conditions)]
sgrnas = ["S" + str(i) for i in range(n_sgrnas)]
replicates = ["R" + str(i) for i in range(n_replicates)]


# In[43]:


combinations = [(g, c, s, r)      for g in genes for c in conditions      for s in sgrnas for r in replicates]


# In[44]:


count_table = pd.DataFrame(
    combinations, columns=["genes", "conditions", "sgrnas", "replicates"])


# In[45]:


sgrna_ids = np.repeat(["S" + str(i)                        for i in range(n_conditions * n_sgrnas * n_genes)], 
                      n_replicates)
count_table.sgrnas = sgrna_ids
condition_ids = np.repeat(["C" + str(i)                            for i in range(n_genes * n_conditions)],
                          n_sgrnas * n_replicates)
count_table.conditions = condition_ids


# In[46]:


le = preprocessing.LabelEncoder()
for i in count_table.columns.values:
    count_table[i] = le.fit_transform(count_table[i])


# In[47]:


beta_tau = .25
l_tau = .25
beta = st.norm.rvs(np.repeat(gamma, n_conditions), beta_tau)
l = st.norm.rvs(0, l_tau, size = n_conditions * n_genes * n_sgrnas)


# In[48]:


data_tau = .25
data = st.norm.rvs(
    l[count_table["sgrnas"]] + beta[count_table["conditions"]],
    data_tau)


# In[49]:


count_table = pd.DataFrame(
    combinations, 
    columns=["gene", "condition", "intervention", "replicate"])

count_table["gamma"] = np.repeat(gamma, count_table.shape[0] / len(gamma))
count_table["beta"] = np.repeat(beta, count_table.shape[0] / len(beta))
count_table["l"] = np.repeat(l, count_table.shape[0] / len(l))
count_table["readout"] = data

sgrna_ids = np.repeat(["S" + str(i) for i in range(n_conditions * n_sgrnas * n_genes)], 
                      n_replicates)
count_table.intervention = sgrna_ids


# In[50]:


count_table


# In[51]:


count_table.to_csv("../data_raw/easy_simulated_data/simulated_data.tsv", index=False, sep="\t")


# In[52]:


networkx.readwrite.edgelist.write_weighted_edgelist(
    G_filtered, "../data_raw/easy_simulated_data/graph.tsv", delimiter="\t")


# In[53]:


data = {    
    "graph": G_filtered,
    "essential_genes": essential_genes,
    "nonessential_genes": nonessential_genes,    
    "gamma_tau": gamma_tau,
    "gamma_tau_non_essential": gamma_tau_non_essential,
    "gamma_essential": gamma_essential,
    "gamma_nonessential": gamma_nonessential,    
    "beta_tau": beta_tau,    
    "beta": beta,
    "l_tau": l_tau,
    "l": l,
    "data_tau": data_tau,
    "data": data,
    "count_table": count_table
}


# In[54]:


with open("../data_raw/easy_simulated_data/data.pickle","wb") as out:
    pickle.dump(data, out)

