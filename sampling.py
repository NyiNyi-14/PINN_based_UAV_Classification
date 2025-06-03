# %% Import Libraries
import numpy as np
import pyDOE
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Sampling Methods
class sampling:
    def __init__(self, para, samples):
        self.para = para
        self.samples = samples
    
    def MC_sampling(self):
        para_MC = {}
        for key, (mean, std) in self.para.items():
            para_MC[key] = np.random.normal(mean, std, self.samples)
        return para_MC
        
    def LHS_sampling(self):
        para_LHS = {}
        num_params = len(self.para) 
        lhs_samples = pyDOE.lhs(num_params, self.samples, criterion='c')
        for i, (key, (mean, std)) in enumerate(self.para.items()):
            para_LHS[key] = stats.norm(loc=mean, scale=std).ppf(lhs_samples[:, i])  # uniform LHS to normal
        return para_LHS
    
    def IM_sampling(self):
        para_IM = {}
        for key, (mean, std) in self.para.items():
            proposal_std = std * 0.5
            para_IM[key] = np.random.normal(mean, proposal_std, self.samples)
        return para_IM
    
    def plot_distributions(samples, method_name):
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))  
        axes = axes.flatten()  

        for i, param in enumerate(samples.keys()):  
            sns.histplot(samples[param], bins=30, kde=True, ax=axes[i], alpha=0.6)
            axes[i].set_title(f"{method_name}: {param} Distribution")
            axes[i].set_xlabel("Parameter Value")
            axes[i].set_ylabel("Frequency")
            axes[i].grid()

        plt.tight_layout()
        plt.show()

# %%
