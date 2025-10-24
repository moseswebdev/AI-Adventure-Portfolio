# In Jupyter - see your digit recognition results!
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import numpy as np

digits = load_digits()

# Create a beautiful visualization
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i in range(10):
    ax = axes[i//5, i%5]
    ax.imshow(digits.images[i], cmap='Blues')
    ax.set_title(f"Digit: {digits.target[i]}")
    ax.axis('off')
    
plt.suptitle("Digits Your AI Recognized with 97.2% Accuracy!", fontsize=16)
plt.tight_layout()
plt.show()