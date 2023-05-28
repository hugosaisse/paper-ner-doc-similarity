# similarity_pairs.py
import numpy as np
import os
import pandas as pd
import itertools
from glob import glob

class SimilarityPairs:
    def __init__(self, vectorized_df=None, sim_matrices_folder=None):
        self.vectorized_df = vectorized_df
        # The folder where the similarity matrices are stored
        self.sim_matrices_folder = sim_matrices_folder
        # The folder where the similarity pairs generated from the similarity matrices will be stored
        self.sim_pairs_folder = "sim_pairs"

    # Identify all CSV files in the "sim_matrices" folder
    def identify_csv_files(self):
        return glob(f"{self.sim_matrices_folder}/*.csv")

    # Read the CSV file as a DataFrame, set the index and column names using the "infracaoId" column
    def read_csv_and_set_index(self, file_path):
        sim_matrix = pd.read_csv(file_path)
        index_col = self.vectorized_df["infracaoId"].tolist()
        sim_matrix.index = index_col
        sim_matrix.columns = index_col
        return sim_matrix

    # Create a new DataFrame containing all two-by-two combinations of infraction IDs and their similarity
    def create_pairs_dataframe(self, sim_matrix):
        infracao_ids = self.vectorized_df["infracaoId"].tolist()
        pairs = list(itertools.combinations(infracao_ids, 2))

        data = []
        for inf_a, inf_b in pairs:
            sim = sim_matrix.loc[inf_a, inf_b]
            data.append({"Inf A": inf_a, "Inf B": inf_b, "Sim": sim})

        return pd.DataFrame(data)

    # Save the DataFrames containing similarity pairs as CSV files in the "sim_pairs" subfolder
    def save_pairs_dataframes(self):
        # Create the "sim_pairs" subfolder if it does not exist
        if not os.path.exists(self.sim_pairs_folder):
            os.makedirs(self.sim_pairs_folder)

        # Iterate through the CSV files in the "sim_matrices" folder
        csv_files = self.identify_csv_files()
        for file_path in csv_files:
            # Read the CSV file, set index and column names
            sim_matrix = self.read_csv_and_set_index(file_path)
            # Create a DataFrame containing infraction pairs and their similarity
            pairs_df = self.create_pairs_dataframe(sim_matrix)
            # Save the pairs DataFrame as a CSV file in the "sim_pairs" folder
            output_file = os.path.join(self.sim_pairs_folder, os.path.basename(file_path))
            pairs_df.to_csv(output_file, index=False)

    # Identify all xlsm files in the "sim_pairs" folder
    def identify_xlsm_files(self):
        return glob(f"{self.sim_pairs_folder}/*.xlsm")

    # Read xlsm files and return a list of DataFrames
    def read_xlsm_files(self):
        xlsm_files = self.identify_xlsm_files()
        dataframes = [pd.read_excel(file, engine="openpyxl") for file in xlsm_files]
        return dataframes

    # Calculate the mean of experts' evaluations and save as a new xlsx file
    def create_mean_experts_evaluation(self, output_file="mean_experts_evaluation.xlsx"):
        if not os.path.isfile(f'{self.sim_pairs_folder}/{output_file}'):
            dataframes = self.read_xlsm_files()
            if not dataframes:
                print("No xlsm files found in the 'sim_pairs' folder.")
                return

            # Concatenate DataFrames and group by 'Inf A' and 'Inf B', calculating the mean of 'Sim' values
            concatenated_df = pd.concat(dataframes)
            mean_evaluations = concatenated_df.groupby(['Inf A', 'Inf B']).agg({'Sim': np.mean}).reset_index()

            # Save the mean evaluations DataFrame as an xlsx file
            output_path = os.path.join(self.sim_pairs_folder, output_file)
            mean_evaluations.to_excel(output_path, index=False, engine="openpyxl")
        else:
            print(f"File '{output_file}' already exists in the {self.sim_pairs_folder} folder.")
