import torch
import torchvision.models as models

# Load the pre-trained VGG model (change the path to your model's location)
model = torch.load('checkpoint/VGG/sr0.00001_threshold_0.01/best.pth')

# Define a function to extract weights, standard deviation, and variance of one layer
def extract_weights_and_stats(layer_name):
    if hasattr(model, layer_name):
        layer = getattr(model, layer_name)
        weights = layer.weight.data
        std = weights.std().item()
        var = weights.var().item()

        return weights, std, var
    else:
        return None

# Example: extract weights and statistics for the 'features.0' layer
layer_name = 'features.0'
weights, std, var = extract_weights_and_stats(layer_name)

# Print or save weights to a text file
if weights is not None:
    print(f"Weights of layer {layer_name}:\n{weights}")
    print(f"Standard Deviation: {std}")
    print(f"Variance: {var}")
else:
    print(f"Layer {layer_name} not found in the model.")

# You can now save the weights to a .txt file if needed.
