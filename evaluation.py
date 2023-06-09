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

# The pre-trained summarization models:
summ_models = ['dominguesm/legal-bert-ner-base-cased-ptbr', 'lisaterumi/postagger-portuguese']
# model2 is the embedding model from OpenAI
model2 = 'text-embedding-ada-002'

similarity_pairs_obj = SimilarityPairs()
# Create a new xlsx file with the mean of experts' evaluations
similarity_pairs_obj.create_mean_experts_evaluation()
# sim_experts is a dictionary of dataframes, where each dataframe corresponds to an expert evaluation
sim_experts = {}
# expert_files is a dictionary of file names, where each file corresponds to an expert evaluation
expert_files = {
    'expert_mean': 'mean_experts_evaluation.xlsx',
    'expert_I': 'similarity_experts_evaluation_I.xlsm',
    'expert_L': 'similarity_experts_evaluation_L.xlsm',
    'expert_S': 'similarity_experts_evaluation_S.xlsm'
}

# Load expert evaluations to the sim_experts dictionary
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

# sim_models is a dictionary of dataframes, where each dataframe converts a model's similarity matrix
# to the form of infraction pair + similarity score
sim_models = {}

for model1 in summ_models:
    # 1) For each embedding model, try loading a pickle file containing a dataframe with embedded (vectorized) infraction text
    for tag, path in zip(['hs', 'if', 'ada'],
                        [model1, model1, f'{model1}/{model2}']
                        ):
        filename = f'./models/{path}/data/vectorized_{tag}_dataset.pkl'
        try:
            with open(filename, 'rb') as f:
                vectorized_df = pickle.load(f)
        except FileNotFoundError:
            # 2) If the file is not present, raise an exception with an error message
            raise FileNotFoundError(f"The file '{filename}' is missing.")

        # Create the similarity pairs using all files contained in the "sim_matrices" folder
        # Each file corresponds to a similarity matrix generated using a different type of summary
        sim_matrices_folder = f'./models/{path}/sim_matrices'
        similarity_pairs_obj = SimilarityPairs(vectorized_df, sim_matrices_folder=sim_matrices_folder, path=f'./models/{path}')
        similarity_pairs_obj.save_pairs_dataframes()

        # sim_pairs_path is the path to the folder containing the similarity pairs generated from the similarity matrices
        sim_pairs_path = f'./models/{path}/sim_pairs'
        # sim_files is a list of file names, where each file contains similarity pairs generated from a similarity matrix
        sim_files = os.listdir(sim_pairs_path)

        # Load similarity matrices to the sim_models dictionary
        for file_name in sim_files:
            if file_name.endswith('.csv'):
                sim_key = path + '/' + file_name[:-4]
                sim_models[sim_key] = pd.read_csv(os.path.join(sim_pairs_path, file_name), index_col=['Inf A', 'Inf B'])

# Instantiate the SimilarityMetrics class
sim_metrics = SimilarityMetrics(sim_experts, sim_models)

# Create the "results" subfolder if it does not exist
if not os.path.exists("results"):
    os.makedirs("results")

# For each expert evaluation, calculate the metrics and save the results to a CSV file
for sim_expert in sim_experts.keys():
    results = sim_metrics.calculate_metrics(sim_expert)
    results.to_csv(f'./results/{sim_expert}_results.csv')
