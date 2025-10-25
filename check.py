from scipy import io
import numpy as np

data = io.loadmat("data/raw_frequency_750.mat")

R = data["R"]
drugs = [d[0] for d in data["drugs"].flatten()]
sideeffects = [s[0] for s in data["sideeffects"].flatten()]

print("Ma trận R:", R.shape)
print("Số thuốc:", len(drugs))
print("Số side effect:", len(sideeffects))

# In thử 5 dòng đầu tiên
for i in range(5):
    print(f"{i+1}. {drugs[i]} → các side effect có nhãn 1:")
    active_effects = [sideeffects[j] for j in np.where(R[i] == 1)[0]]
    print("   ", active_effects[:10])  # in tối đa 10 hiệu ứng đầu
