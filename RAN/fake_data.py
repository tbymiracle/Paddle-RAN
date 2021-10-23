import numpy as np
fake_data = np.random.rand(64, 3, 32, 32).astype(np.float32)
fake_label = np.ones(64,dtype=int)
print(fake_label)
np.save("fake_data.npy", fake_data)
np.save("fake_label.npy", fake_label)