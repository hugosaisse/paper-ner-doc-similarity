"""
This file is used to evaluate the similarity pairs generated using the full text
and the different types of summaries.
"""

import pickle
import pandas as pd
from similarity_pairs.similarity_pairs import SimilarityPairs

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
