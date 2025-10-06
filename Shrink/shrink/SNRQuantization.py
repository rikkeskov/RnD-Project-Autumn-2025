import numpy as np
import pandas as pd
import math

def Encoding(data, beta=None, snr=None):
    if snr is None:
        eps = 2 ** beta
        quantized_data = np.round(data / eps) * eps
    else:
        eps, current_snr, quantized_data = quantize(data, snr)

    return eps, current_snr, quantized_data

def quantize(a, target_snr):
    beta = init_beta(a)
    # print(f"init_beta: {beta}")
    pre_snr = float('inf')
    iteration, max_iterations = 0, 50  # 设置最大迭代次数，防止死循环
    while iteration<max_iterations:
        iteration += 1
        count, current_snr = quantize_with_beta(a, beta)
        # print(f"current_snr = {current_snr}")
        if current_snr < target_snr:
            if(pre_snr!=float('inf')):
                beta -= 1
                _, current_snr = quantize_with_beta(a, beta)
                eps = 2 ** beta
                break
            else:
                beta -= 2
                _, current_snr = quantize_with_beta(a, beta)
                eps = 2 ** beta
                break

        beta += 1
        pre_snr = current_snr

    return eps,current_snr,np.round(a / eps) * eps

def init_beta(a):
    power = np.mean(a ** 2) * 1e-6
    return int(math.floor(0.5 * math.log2(power)) + 1)


def quantize_with_beta(a, beta):
    eps = 2 ** beta
    signal_power = np.sum(a ** 2)
    noise_power = 0
    for value in a:
        quantized_value = round(value / eps) * eps
        error = value - quantized_value
        noise_power += error ** 2
    snr = 10 * math.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    # print(f"beta: {beta}, noise_power: {noise_power}, snr: {noise_power}, ", end="")
    return len(a), snr

if __name__ =="__main__":

    path =  "/home/guoyou/OutlierDetection/TSB-UAD/data/OPPORTUNITY/"
    filenames = ["S1-ADL1.test.csv@16.out"]
    data = pd.read_csv(path+filenames[0], header=None)
    values = data[0].values
    print(values)
    target_snr, beta = 25, 4 
    eps, current_snr,quantized_data = Encoding(values, beta=beta, snr=target_snr)
    print(f"\ntarget_snr = {target_snr}, eps = {eps}, current_snr = {current_snr}")


















# import numpy as np 
# import pandas as pd
# import math

# def Encoding(data, beta=None, snr=None):
#     if snr is None:
#         eps = 2 ** beta
#         quantized_data = np.round(data / eps) * eps
#     else:
#         eps, current_snr, quantized_data = quantize(data, snr)

#     return eps, current_snr, quantized_data

# def quantize(a, target_snr):
#     beta = init_beta(a)
#     while True:
#         count, current_snr = quantize_with_beta(a, beta)
#         if current_snr < target_snr:
#             break
#         beta += 1
#     beta -= 1
#     _, current_snr = quantize_with_beta(a, beta)
#     eps = 2 ** beta
#     return eps,current_snr,np.round(a / eps) * eps

# def init_beta(a):
#     power = np.mean(a ** 2) * 1e-6
#     return int(math.floor(0.5 * math.log2(power)) + 1)

# def quantize_with_beta(a, beta):
#     eps = 2 ** beta
#     signal_power = np.sum(a ** 2)
#     noise_power = 0
#     for value in a:
#         quantized_value = round(value / eps) * eps
#         error = value - quantized_value
#         noise_power += error ** 2
#     snr = 10 * math.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
#     return len(a), snr

# if __name__ =="__main__":

#     path = "/home/guoyou/OutlierDetection/TSB-UAD/data/Genesis/"
#     filenames = ["genesis-anomalies.test.csv@6.out"]

#     data = pd.read_csv(path+filenames[0], header=None)
#     values = data[1].values
#     beta = 4 
#     target_snr = 25
#     eps, current_snr,quantized_data = Encoding(values, beta=beta, snr=target_snr)
#     print(f"target_snr = {target_snr}, eps = {eps}, current_snr = {current_snr}")
#     # print(eps)
#     # print(current_snr)
#     # print(data)
#     # print(quantized_data)