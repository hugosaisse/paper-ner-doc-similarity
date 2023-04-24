import os
import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityMatrixCalculator:
    def __init__(self, vectorized_df, columns):
        self.vectorized_df = vectorized_df
        self.columns = columns
        self.similarity_matrices = {}
        
    def calculate_similarity_matrices(self):
        for column in self.columns:
            embeddings = torch.stack(self.vectorized_df[column].tolist()).numpy()
            # Calculate similarity matrix for each document with all other documents, including itself
            similarity_matrix = cosine_similarity(embeddings, embeddings)
            self.similarity_matrices[column] = similarity_matrix

    def save_similarity_matrices(self, output_folder="sim_matrices"):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for column, matrix in self.similarity_matrices.items():
            output_file = os.path.join(output_folder, f"{column}_similarity.csv")
            pd.DataFrame(matrix).to_csv(output_file, index=False)

