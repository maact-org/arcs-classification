import matplotlib.pyplot as plt
import pandas as pd
#%%
memory=pd.read_csv('hedonometer/data/memory.csv')
#%%
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(memory.index,memory.train_loss)
plt.show()
#%%
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(memory.index,memory.val_loss)
plt.show()
#%%
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(memory.index,memory.train_acc)
plt.show()
#%%
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(memory.index,memory.val_acc)
plt.show()

