import matplotlib.pyplot as plt
import os
import pandas as pd

model_dir = 'results/testdrop05'

df = pd.read_csv(os.path.join(model_dir, 'train_losses.log'))

# Loss plot
df['Train Loss'].plot(color='blue', label='Train')
df['Valid Loss'].plot(color='orange')
plt.xlabel('Epoch')
plt.ylabel('BCE Loss')
plt.legend()

# AUROC plot
plt.figure()
df['AUROC'].plot(color='orange', label='Valid')
plt.xlabel('Epoch')
plt.ylabel('AUROC')
plt.legend()
plt.show()
