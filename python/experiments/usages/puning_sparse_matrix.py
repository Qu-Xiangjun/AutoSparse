import torch
from transformers import BertModel
import matplotlib.pyplot as plt

model = BertModel.from_pretrained("madlag/bert-base-uncased-squad1.1-block-sparse-0.07-v1")

state_dict = model.state_dict()

for key, value in state_dict.items():
    if 'weight' in key:
        non_zero_count = torch.nonzero(value).size(0)
        total_elements = value.numel()
        sparsity = non_zero_count / total_elements
        if sparsity < 0.05:
            print(f"{key} - Sparsity: {sparsity:.8f}")
            
            processed_value = torch.where(abs(value) < 0.0001, torch.tensor(0), torch.tensor(1))

            plt.imshow(processed_value.detach().cpu().numpy(), cmap='binary', interpolation='none')
            plt.show()
            plt.axis('off')
            plt.savefig(f"{key}_sparse_image.png", bbox_inches='tight', pad_inches=0)
