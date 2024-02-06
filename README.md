<h1 align="center"> 🌲 MetaTree 🌲 </h1>
<p align="center"> <b>Learning a Decision Tree Algorithm with Transformers</b>  
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-mit-blue.svg">
  <img src="https://img.shields.io/badge/python-3.7+-blue">
  <img src="https://img.shields.io/pypi/v/metatreelib?color=green">  
</p>  

<p align="center"> MetaTree is a transformer-based decision tree algorithm. It learns from classical decision tree algorithms (greedy algorithm CART, optimal algorithm GOSDT), for better generalization capabilities.
</p>

## Quickstart -- use MetaTree to generate decision tree models

Model is avaliable at https://huggingface.co/yzhuang/MetaTree

1. Install `metatreelib`:

```bash
pip install metatreelib
# Alternatively,  
# clone then pip install -e .
# pip install git+https://github.com/EvanZhuang/MetaTree
```

2. Use MetaTree on your datasets to generate a decision tree model
 
```python
from metatree.model_metatree import LlamaForMetaTree as MetaTree
from metatree.decision_tree_class import DecisionTree, DecisionTreeForest
from metatree.run_train import preprocess_dimension_patch
from transformers import AutoConfig
import imodels # pip install imodels 

# Initialize Model
model_name_or_path = "yzhuang/MetaTree"

config = AutoConfig.from_pretrained(model_name_or_path)
model = MetaTree.from_pretrained(
    model_name_or_path,
    config=config,
)   

# Load Datasets
train_X, train_Y, test_X, test_Y = your_data_function()

input_x = torch.tensor(train_X, dtype=torch.float32)
input_y = torch.nn.functional.one_hot(torch.tensor(train_y)).float()

# Generate a decision tree with MetaTree
decision_tree_forest = DecisionTreeForest()
model.depth = 2
outputs = model.generate_decision_tree(input_x, input_y, depth=model.depth)
decision_tree_forest.add_tree(DecisionTree(auto_dims=outputs.metatree_dimensions, auto_thresholds=outputs.tentative_splits, input_x=input_x, input_y=input_y, depth=model.depth))

print("Decision Tree Features: ", [x.argmax(dim=-1) for x in outputs.metatree_dimensions])
print("Decision Tree Threasholds: ", outputs.tentative_splits)
```

3. Inference with the decision tree model

```python
tree_pred = decision_tree_forest.predict(torch.tensor(test_X, dtype=torch.float32))

accuracy = accuracy_score(test_y, tree_pred.argmax(dim=-1).squeeze(0))
print("MetaTree Test Accuracy: ", accuracy)
```

## Example Usage

We show a complete example of using MetaTree at [notebook](examples/example_usage.ipynb)

## Questions?

If you have any questions related to the code or the paper, feel free to reach out to us at y5zhuang@ucsd.edu.