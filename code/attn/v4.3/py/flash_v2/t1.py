import torch


import numpy as np



def np_uint16_to_bfloat16_tensor(arr_uint16: np.ndarray):
    assert arr_uint16.dtype == np.uint16
    arr_uint8 = arr_uint16.view(np.uint8)
    t = torch.from_numpy(arr_uint8).view(torch.int16)
    return t.view(torch.bfloat16)



np_arr = np.array([0x3E00, 0x3F80, 0xC000], dtype=np.uint16)
bf16_tensor = np_uint16_to_bfloat16_tensor(np_arr)
print(bf16_tensor)        # tensor([ 0.1250,  1.0000, -2.0000], dtype=torch.bfloat16)
print(bf16_tensor.dtype)  # torch.bfloat16
