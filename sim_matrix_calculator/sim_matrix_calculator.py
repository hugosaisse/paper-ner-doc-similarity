import os
import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityMatrixCalculator:
    def __init__(self, vectorized_df, columns, model_path):
        self.vectorized_df = vectorized_df
        self.columns = columns
        self.similarity_matrices = {}
        self.model_path = model_path
        
    def calculate_similarity_matrices(self, embedding_type):
        if embedding_type == "tensors":
            self._calculate_similarity_matrices_tensors()
        elif embedding_type == "lists":
            self._calculate_similarity_matrices_lists()
        else:
            raise Exception("Invalid embedding type. Must be 'tensors' or 'lists'.")

    def _calculate_similarity_matrices_tensors(self):
        for column in self.columns:
            # Squeeze the tensor to remove any extra dimension
            # Inference API results in 2-dimensional tensors, e.g. (1, 768) -> (768,)
            self.vectorized_df[column] = self.vectorized_df[column].apply(lambda x: x.squeeze())
            # Convert tensor to numpy array
            embeddings = torch.stack(self.vectorized_df[column].tolist()).numpy()
            # Calculate similarity matrix for each document with all other documents, including itself
            similarity_matrix = cosine_similarity(embeddings, embeddings)
            self.similarity_matrices[column] = similarity_matrix

    def _calculate_similarity_matrices_lists(self):
        for column in self.columns:
            # Convert column to list of lists
            embeddings = self.vectorized_df[column].tolist()
            # Calculate similarity matrix for each document with all other documents, including itself
            similarity_matrix = cosine_similarity(embeddings, embeddings)
            self.similarity_matrices[column] = similarity_matrix

    def save_similarity_matrices(self, subfolder="sim_matrices"):
        output_folder = os.path.join('./models', self.model_path, subfolder)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for column, matrix in self.similarity_matrices.items():
            output_file = os.path.join(output_folder, f"{column}_similarity.csv")
            pd.DataFrame(matrix).to_csv(output_file, index=False)

