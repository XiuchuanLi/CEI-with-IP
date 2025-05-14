import numpy as np
from algorithm.src.utils import independence


def SelectPdf(Num,data_type):
    if data_type == "uniform":
        noise = np.random.uniform(-1, 1, size=Num)

    elif data_type == "laplace":
        noise =np.random.laplace(0, 1, size=Num)

    else: #gauss
        noise = np.random.normal(0, 1, size=Num)

    return noise


def normalize(data):
    data -= np.mean(data)
    data /= np.std(data)
    return data


for distribution in ['laplace', 'uniform']:
    for Num in [500,1000,2000,5000,10000,15000,20000,25000]:
        noises = []
        for i in range(10):
            print(i)
            while True:
                new_noise = normalize(SelectPdf(Num, distribution))
                if np.all(np.array([np.abs(np.corrcoef(new_noise, noise)[0, 1]) < 0.02 for noise in noises])) \
                    and np.all(np.array([independence(new_noise, noise, 0.2)[0] for noise in noises])):
                    # Make sure that the exogenous noises are independent and uncorrelated to each other
                    noises.append(new_noise)
                    break
        noises = np.stack(noises, axis=0)
        np.save(f'{distribution}_noises/noise_{Num}.npy', noises)