import numpy as np
arr = [np.random.normal(0,1,(4,)) for i in range(10)]
print(np.stack(arr, axis=1))