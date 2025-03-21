import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.positional_encoding import PositionalEncoding
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils.helpers import show_example

def example_positional():
    pe = PositionalEncoding(20, 0)
    y = pe.forward(torch.zeros(1, 100, 20))

    data = pd.concat(
        [
            pd.DataFrame(
                {
                    "embedding": y[0, :, dim],
                    "dimension": dim,
                    "position": list(range(100)),
                }
            )
            for dim in [4, 5, 6, 7]
        ]
    )
    # Vẽ biểu đồ bằng Seaborn
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x='position', y='embedding', hue='dimension', palette='tab10')
    
    plt.xlabel('Position')
    plt.ylabel('Embedding')
    plt.title('Positional Encoding Example')
    plt.grid(True)
    return plt

# Gọi và hiển thị biểu đồ
plot = show_example(example_positional)
if plot:
    plot.show()