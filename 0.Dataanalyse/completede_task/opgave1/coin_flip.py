import random 
import matplotlib.pyplot as plt
import numpy as np

def coin_toss_sequence(n):

    krone_count = 0
    plat_count = 0
    prob_history_krone = []
    prob_history_plat = []
    
    for i in range(1, n + 1):
        if random.randint(0, 1) == 0:
            krone_count += 1
        else:
            plat_count += 1

        prob_history_krone.append(krone_count / i)
        prob_history_plat.append(plat_count / i)
    
    return prob_history_krone,prob_history_plat

# Simuler mange kast
flipsKroneX = 100000
flipsKrone = 1000
flipsPlat  = 100
prob_10000_krone, prob_10000_plat = coin_toss_sequence(flipsKroneX)
prob_1000_krone, prob_1000_plat = coin_toss_sequence(flipsKrone)
prob_100_krone, prob_100_plat = coin_toss_sequence(flipsPlat)

# Beregn plat pr 100 kast 
avg = np.mean(prob_100_plat)
print(avg * 100 , "avg")


plt.figure(figsize=(8, 6))
plt.plot(range(1, flipsKrone + 1), prob_1000_krone, label='Krone (1,000 flips)', color='blue', alpha=0.7)
plt.plot(range(1, flipsKroneX + 1), prob_10000_krone, label='Krone (1,0000 flips)', color='red', alpha=0.5)
plt.axhline(y=0.5, color='black', linestyle='--', label='Expected Probability 50%')

plt.legend()
plt.grid(True)
plt.show() 

