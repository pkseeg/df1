from sklearn.metrics import precision_recall_fscore_support
import numpy as np

from sentence_transformers import SentenceTransformer
from tte_depth import StatDepth


class DF1:
    def __init__(self, source_texts, target_texts, model_card):
        # FIXME add documentation -- I need to explain that the source_texts should be sampled, and the target_texts too
        self.source_median, self.target_depths = self._depth_scores(source_texts, target_texts, model_card)

    def _depth_scores(self, source_texts, target_texts, model_card):
        # Load sentence transformer model
        model = SentenceTransformer(model_card)

        # encode source and target texts using the sentence transformer model
        encoded_source = model.encode(source_texts)
        encoded_target = model.encode(target_texts)

        # depth_rank_test returns depth scores for each corpus, along with a Q estimate, W test statistic from the Wilcoxon Rank Sum Test, and an associated p-value
        d = StatDepth()
        source_depths, target_depths, _, _, _ = d.depth_rank_test(encoded_source, encoded_target)

        # calculate the median depth of the source_texts
        source_median = np.median(source_depths)
        return source_median, target_depths
    
    def _ranks(self):
        # Create a sorted version of the list
        sorted_lst = sorted(self.target_depths)
        
        # Create a dictionary to store ranks
        rank_dict = {value: index for index, value in enumerate(sorted_lst)}
        
        # Create a new list containing ranks of elements in the original list
        result = [rank_dict[value] for value in self.target_depths]
    
        return result
    
    def precision_recall_fscore_support(self, y_true, y_pred, estimated_overlap = 0.2, **kwargs):
        assert len(y_pred) == len(self.target_depths),  "y_pred does not match the number of target texts!"
        # calculate the distances between the depth of the source median and the depth of each target text
        dists = [self.source_median - x for x in self.target_depths]
        percentile = np.percentile(dists, estimated_overlap)
        zeroed_out = [x if x > percentile else 0 for x in dists] # only take those about some estimate overlap threshold
        self.target_weights = [d/sum(zeroed_out) for d in zeroed_out]
        #self.target_weights = self._ranks()
        return precision_recall_fscore_support(y_true, y_pred, sample_weight=self.target_weights, **kwargs)