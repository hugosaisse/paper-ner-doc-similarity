"""
This file is used to evaluate the similarity pairs generated using the full text
and the different types of summaries.
"""
import os
import pickle
import pandas as pd
from openpyxl import load_workbook
from similarity_pairs.similarity_pairs import SimilarityPairs
from similarity_metrics.similarity_metrics import SimilarityMetrics

# 1) Try loading a pickle file containing a dataframe called "vectorized_df"
try:
    with open('./data/vectorized_dataset.pkl', 'rb') as f:
        vectorized_df = pickle.load(f)
except FileNotFoundError:
    # 2) If the file is not present, raise an exception with an error message
    raise FileNotFoundError("The file './data/vectorized_dataset.pkl' is missing.")

# Create the similarity pairs using all files contained in the "sim_matrices" folder
# Each file corresponds to a similarity matrix generated using a different type of summary
similarity_pairs_obj = SimilarityPairs(vectorized_df)
similarity_pairs_obj.save_pairs_dataframes()

# Create a new xlsx file with the mean of experts' evaluations
similarity_pairs_obj.create_mean_experts_evaluation()

# Load expert evaluations
# sim_experts is a dictionary of dataframes, where each dataframe corresponds to an expert evaluation
sim_experts = {}
# expert_files is a dictionary of file names, where each file corresponds to an expert evaluation
expert_files = {
    'expert_mean': 'mean_experts_evaluation.xlsx',
    'expert_I': 'similarity_experts_evaluation_I.xlsm',
    'expert_L': 'similarity_experts_evaluation_L.xlsm',
    'expert_S': 'similarity_experts_evaluation_S.xlsm'
}

for expert_name, file_name in expert_files.items():
    if file_name.endswith('.xlsx') or file_name.endswith('.xlsm'):
        wb = load_workbook(filename=os.path.join('sim_pairs', file_name), read_only=True, data_only=True)
        ws = wb.active
        data = ws.values
        columns = next(data)[0:]
        df = pd.DataFrame(data, columns=columns).set_index(['Inf A', 'Inf B'])
        if "Num" in df.columns:
            df = df.drop(columns=['Num'])
        sim_experts[expert_name] = df.copy(deep=True)

# Load summary of similarity matrices
# sim_models is a dictionary of dataframes, where each dataframe corresponds to a similarity matrix in the form of infraction pairs
sim_models = {}
sim_files = os.listdir('sim_pairs')

for file_name in sim_files:
    if file_name.endswith('.csv'):
        sim_key = file_name[:-4]
        sim_models[sim_key] = pd.read_csv(os.path.join('sim_pairs', file_name), index_col=['Inf A', 'Inf B'])

# Instantiate the SimilarityMetrics class
sim_metrics = SimilarityMetrics(sim_experts, sim_models)

# Create the "sim_pairs" subfolder if it does not exist
if not os.path.exists("results"):
    os.makedirs("results")

# For each expert evaluation, calculate the metrics and save the results to a CSV file
for sim_expert in sim_experts.keys():
    results = sim_metrics.calculate_metrics(sim_expert)
    results.to_csv(f'./results/{sim_expert}_results.csv')
