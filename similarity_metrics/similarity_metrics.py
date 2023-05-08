import pandas as pd
import numpy as np
from tqdm import tqdm

class SimilarityMetrics:
    def __init__(self, sim_experts, sim_models):
        # sim_experts: a dictionary of infraction pairs with similarity scores given by experts
        self.sim_experts = sim_experts
        # sim_models: a dictionary of infraction pairs with similarity scores given by a model
        self.sim_models = sim_models

    def top_n_similar(self, n, infraction_id, sim_key, sim_column='Sim', threshold=0.0):
        if sim_key in self.sim_experts:
            sim_dataframe = self.sim_experts[sim_key]
        else:
            sim_dataframe = self.sim_models[sim_key]
        
        # how to loc using MultiIndex if infraction_id may be present at any level?
        # sim_dataframe has a MultiIndex with levels 'Inf A' and 'Inf B'
        try:
            level1 = sim_dataframe.xs(infraction_id, level='Inf A')
        except:
            level1 = pd.DataFrame()
        try:
            level2 = sim_dataframe.xs(infraction_id, level='Inf B')
        except:
            level2 = pd.DataFrame()
        
        sim_array = pd.concat([level1, level2])
        sim_array = sim_array[sim_array[sim_column] > threshold]
        sim_array = sim_array.nlargest(n, sim_column)
        
        return set(sim_array.index)

    def calculate_metrics(self, sim_expert):
        # Find unique gold standard infraction pairs from all of the expert evaluation files
        gold_std_infraction_pairs = set()
        for exp_eval in self.sim_experts.values():
            gold_std_infraction_pairs = gold_std_infraction_pairs.union(exp_eval.index.unique())
        
        # Find unique gold standard infractions from all of the expert evaluation files
        gold_std_infractions = set()
        for pair in gold_std_infraction_pairs:
            gold_std_infractions = gold_std_infractions.union(pair)

        results = pd.DataFrame(index=self.sim_models.keys(), columns=['mAR', 'recall@5', 'mAP', 'precision@5'])

        for sim_model in self.sim_models.keys():
            avg_recall_list = []
            avg_precision_list = []

            for n in tqdm(range(1, len(gold_std_infractions) + 1)):
                recall_list = []
                precision_list = []

                for infraction in gold_std_infractions:
                    # top_n_similar_model: top n similar infractions given by the model
                    top_n_similar_model = self.top_n_similar(n, infraction, sim_model)
                    # top_n_similar_expert: top n similar infractions given by an expert (or mean of experts)
                    top_n_similar_expert = self.top_n_similar(n, infraction, sim_expert)
                    # total_similar_expert: all similar infractions given by an expert (or mean of experts)
                    total_similar_expert = self.top_n_similar(50, infraction, sim_expert, threshold=0.0)

                    # recall: how many relevant infractions are retrieved?
                    # recall: (relevant documents, retrieved documents)/(relevant documents)
                    # precision: how many retrieved infractions are relevant?
                    # precision: (relevant documents, retrieved documents)/(retrieved documents)
                    num = len(top_n_similar_model.intersection(total_similar_expert))
                    #recall_den = len(top_n_similar_expert)
                    recall_den = len(total_similar_expert)
                    precision_den = len(top_n_similar_model)

                    # recall_den == 0 means that there are no similar infractions given by an expert
                    # in this case, recall is 1 because the model has no similar infractions to compare to
                    if recall_den == 0:
                        recall = 1
                    else:
                        recall = num / recall_den

                    # precision_den == 0 means that there are no similar infractions given by the model
                    # if top_n_similar_expert is not empty, then precision is 0 because the model wasn't able to find similar infractions
                    if precision_den == 0 and len(top_n_similar_expert) != 0:
                        precision = 0
                    # if top_n_similar_expert is empty, then precision is 1 because there were no similar infractions to find
                    elif precision_den == 0 and len(top_n_similar_expert) == 0:
                        precision = 1
                    else:
                        precision = num / precision_den

                    recall_list.append(recall)
                    precision_list.append(precision)

                avg_recall = np.mean(recall_list)
                avg_precision = np.mean(precision_list)
                avg_recall_list.append(avg_recall)
                avg_precision_list.append(avg_precision)

                if n == 5:
                    results.at[sim_model, 'recall@5'] = avg_recall
                    results.at[sim_model, 'precision@5'] = avg_precision

            mean_avg_recall = np.mean(avg_recall_list)
            mean_avg_precision = np.mean(avg_precision_list)
            results.at[sim_model, 'mAR'] = mean_avg_recall
            results.at[sim_model, 'mAP'] = mean_avg_precision

        results['f1'] = (2 * results['mAR'] * results['mAP']) / (results['mAR'] + results['mAP'])

        return results