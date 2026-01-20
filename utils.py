# coding: utf-8

# In[1]:


import os


DATA_FOLDER = 'data'
TEST_TCGA_DATA_FOLDER = os.path.join(DATA_FOLDER, 'TCGA_test_data')
RAW_BOTH_DATA_FOLDER = os.path.join(DATA_FOLDER, 'CTRP_GDSC_data')
DRUG_DATA_FOLDER = os.path.join(DATA_FOLDER, 'drug_data')
CCLE_RAW_DATA_FOLDER = os.path.join(DATA_FOLDER, 'CCLE_data')

CCLE_SCREENING_DATA_FOLDER = os.path.join(CCLE_RAW_DATA_FOLDER, 'drug_screening_matrix_ccle.tsv')
BOTH_SCREENING_DATA_FOLDER = os.path.join(RAW_BOTH_DATA_FOLDER, 'drug_screening_matrix_gdsc_ctrp.tsv')

CCLE_FOLDER = os.path.join(DATA_FOLDER, 'CCLE')

TCGA_DATA_FOLDER = os.path.join(DATA_FOLDER, 'TCGA_data')
TCGA_SCREENING_DATA = os.path.join(TCGA_DATA_FOLDER, 'TCGA_screening_matrix.tsv')

MODEL_FOLDER = os.path.join(DATA_FOLDER, 'model')

BUILD_SIM_MATRICES = True  # Make this variable True to build similarity matrices from raw data
SIM_KERNEL = {'cell_exp': ('euclidean', 0.01), 'cell_mut': ('jaccard', 1), 'drug_desc': ('euclidean', 0.001), 'drug_finger': ('euclidean', 0.001)}
SAVE_MODEL = False  # Change it to True to save the trained model
VARIATIONAL_AUTOENCODERS = False
# DATA_MODALITIES=['cell_exp','cell_mut', 'drug_desc', 'drug_finger'] # Change this list to only consider specific data modalities
DATA_MODALITIES = ['cell_exp', 'cell_mut', 'drug_desc', 'drug_finger']
RANDOM_SEED = 42  # Must be used wherever can be used


# In[ ]:




