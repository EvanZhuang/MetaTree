import torch
import torch.nn as nn
import copy
from typing import List, Optional, Tuple, Union, Any

EPS = 1e-5

class DecisionTree(nn.Module):
    def __init__(self, auto_dims: List[torch.LongTensor], auto_thresholds: List[torch.FloatTensor], input_x: torch.FloatTensor, input_y: torch.FloatTensor, depth: int = 2):
        super(DecisionTree, self).__init__()
        self.depth = depth
        self.auto_dims = auto_dims
        self.auto_thresholds = auto_thresholds
        self.leaf_node_labels = [torch.zeros_like(input_y).float() for _ in range(2**self.depth)]
        # okay, we need to record the label for each leaf node
        self._build_tree(input_x, input_y)

        assert len(self.auto_dims) == 2**self.depth - 1, "auto_dims should be a list of length 2**depth - 1"

    def _find_parent(self, idx):
        return (idx-1) // 2

    def _find_path(self, idx):
        # idx is the index of the leaf node
        # return the path from the leaf node to the root, excluding the root
        if idx == 0:
            return []
        return [idx] + self._find_path((idx - 1)// 2)

    def _build_tree(self, x: torch.FloatTensor, y: torch.FloatTensor):
        # Traverse the decision tree and return labeled leaf nodes
        if len(x.shape) <= 2:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        bs, input_len, input_dim = x.shape

        decision_lst = []
        generate_status_lst = [torch.ones((bs, input_len), device=x.device, dtype=x.dtype)]

        # Generate Status for Each Data Point
        for ctr in range(2**self.depth - 1):
            t_status = generate_status_lst.pop(0)
            split_idx = self.auto_dims[ctr].argmax(dim=-1)
            tentative_split = self.auto_thresholds[ctr]
            tentative_x = torch.gather((x + EPS), dim=-1, index=split_idx.view(bs,1,1).expand(bs,input_len,1)).squeeze(-1)

            decision = (tentative_x < tentative_split).float() * t_status
            decision_lst.append(decision)
            generate_status_lst.append(t_status * decision)
            generate_status_lst.append(t_status * (1-decision))

        # Generate Predictions
        y_pred = torch.zeros_like(y).float()
        # iterate over leaf nodes
        for tree_ctr in range(2**self.depth-1, 2**(self.depth+1)-1):
            path = self._find_path(tree_ctr)
            # decision is binary, 1 if the data point is in the leaf node, 0 otherwise
            decision = self._get_decision(decision_lst, path).unsqueeze(-1) # BS, Seq, 1
            node_pred = (y * decision).sum(dim=1, keepdim=True) / (decision.sum(dim=1, keepdim=True) + EPS) # BS, 1, Nclass
            #node_pred = torch.nn.functional.one_hot(node_pred.argmax(dim=-1), num_classes=y.shape[-1]).float() 
            self.leaf_node_labels[tree_ctr - 2**self.depth + 1] = node_pred
        return

    def _get_decision(self, decision_lst, path):
        final_decision = torch.ones_like(decision_lst[0])

        for idx in path:
            if idx % 2 == 1: #Last split = TRUE
                final_decision = final_decision * decision_lst[self._find_parent(idx)]
            else: #Last split = FALSE
                final_decision = final_decision * (1 - decision_lst[self._find_parent(idx)])
        return final_decision

    def forward(self, input_x: torch.FloatTensor, input_y: torch.FloatTensor):
        return self.predict(x=input_x, y=input_y)

    def predict(self, x: torch.FloatTensor):
        # Traverse the decision tree and return predictions
        if len(x.shape) <= 2:
            x = x.unsqueeze(0)
        bs, input_len, input_dim = x.shape

        decision_lst = []
        generate_status_lst = [torch.ones((bs, input_len), device=x.device, dtype=x.dtype)]

        # Generate Status for Each Data Point
        for ctr in range(2**self.depth - 1):
            t_status = generate_status_lst.pop(0)
            split_idx = self.auto_dims[ctr].argmax(dim=-1)
            tentative_split = self.auto_thresholds[ctr]
            tentative_x = torch.gather((x + EPS), dim=-1, index=split_idx.view(bs,1,1).expand(bs,input_len,1)).squeeze(-1)

            decision = (tentative_x < tentative_split).float() * t_status
            decision_lst.append(decision)
            generate_status_lst.append(t_status * decision)
            generate_status_lst.append(t_status * (1-decision))

        # Generate Predictions
        y_pred = 0
        # iterate over leaf nodes
        for tree_ctr in range(2**self.depth-1, 2**(self.depth+1)-1):
            path = self._find_path(tree_ctr)
            # decision is binary, 1 if the data point is in the leaf node, 0 otherwise
            decision = self._get_decision(decision_lst, path).unsqueeze(-1) # BS, Seq, 1
            y_pred += decision * self.leaf_node_labels[tree_ctr - 2**self.depth + 1].unsqueeze(0)
            
            #((y * decision).sum(dim=1, keepdim=True) / (decision.sum(dim=1, keepdim=True) + EPS)) # BS, Seq, Nclass
        return y_pred.squeeze(0)


class DecisionTreeForest(nn.Module):
    def __init__(self) -> None:
        super(DecisionTreeForest, self).__init__()
        self.trees = nn.ModuleList()
    
    def add_tree(self, tree: DecisionTree) -> None:
        self.trees.append(tree)

    def forward(self, input_x: torch.FloatTensor, hard: bool = False):
        return self.predict(x=input_x, hard=hard)
    
    def predict(self, x: torch.FloatTensor, hard: bool = False):
        # Traverse each tree of the forest and return predictions
        forest_pred = 0
        for tree_ctr in range(len(self.trees)):
            y_pred = self.trees[tree_ctr].predict(x)
            if hard:
                y_pred = torch.argmax(y_pred, dim=-1)
                y_pred = torch.nn.functional.one_hot(y_pred, num_classes=y_pred.shape[-1]).float()
            forest_pred += y_pred
        return forest_pred / len(self.trees)