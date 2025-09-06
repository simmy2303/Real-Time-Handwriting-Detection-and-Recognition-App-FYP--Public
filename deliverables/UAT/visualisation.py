import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the data
df = pd.read_csv('UAT_Responses.csv')

# 2. Clean up column names to remove any non-printable chars
df.columns = df.columns.str.replace('\u00A0', ' ')  \
                         .str.strip()

# 3. Select exactly the six question columns (by index or name)
#    Here we assume they are the last six columns in the CSV:
questions = df.columns[-6:].tolist()

# 4. Flatten all scores into one array
all_scores = np.concatenate([
    pd.to_numeric(df[q], errors='coerce').dropna().values
    for q in questions
])

# 5. Compute mean and standard deviation
mu = all_scores.mean()
sigma = all_scores.std(ddof=1)

# 6. Plot histogram + bell curve
x = np.linspace(1, 5, 200)
pdf = (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x-mu)/sigma)**2)

plt.figure(figsize=(8,4))
plt.hist(all_scores, bins=np.arange(0.5,6,1), density=True,
         alpha=0.6, edgecolor='black', label='All UAT Scores')
plt.plot(x, pdf, 'r-', lw=2, label=f'Normal fit (μ={mu:.2f}, σ={sigma:.2f})')
plt.xticks([1,2,3,4,5])
plt.xlabel('Rating')
plt.ylabel('Density')
plt.title('Aggregate UAT Ratings with Bell Curve Fit')
plt.legend()
plt.show()

