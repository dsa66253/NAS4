import numpy as np
path = "./alpha_pdart_nodrop"
try:
    for i in range(1, 45):
        print(path+"/alpha_prob_0_"+str(i)+".npy")
        print(np.load(path+"/alpha_prob_0_"+str(i)+".npy"))
except:
    print("END")