import torch


import numpy as np





N = 1024*1 + 1 # token length

d = 64 # d_model

Br = 8
Bc = 16


#Q_mat = torch.rand((N, d))
#K_mat = torch.rand((N, d))
#V_mat = torch.rand((N, d))


def np_uint16_to_bfloat16_tensor(arr_uint16: np.ndarray):
    assert arr_uint16.dtype == np.uint16
    arr_uint8 = arr_uint16.view(np.uint8)
    t = torch.from_numpy(arr_uint8).view(torch.int16)
    return t.view(torch.bfloat16)



q_uint16 = np.load("../npy_save_single_head/q_uint16.npy")
Q_mat = np_uint16_to_bfloat16_tensor(q_uint16)

k_uint16 = np.load("../npy_save_single_head/k_uint16.npy")
K_mat = np_uint16_to_bfloat16_tensor(k_uint16)

v_uint16 = np.load("../npy_save_single_head/v_uint16.npy")
V_mat = np_uint16_to_bfloat16_tensor(v_uint16)

o_uint16 = np.load("../npy_save_single_head/o_uint16.npy")
O_mat = np_uint16_to_bfloat16_tensor(o_uint16)



QQ_mat = torch.rand(13, 16, 1025, 64, dtype=torch.bfloat16)
KK_mat = torch.rand(13, 16, 1025, 64, dtype=torch.bfloat16)
VV_mat = torch.rand(13, 16, 1025, 64, dtype=torch.bfloat16)
OO_mat = torch.rand(13, 16, 1025, 64, dtype=torch.bfloat16)

for b in range(13):
    for i in range(16):
        QQ_mat[b, i, :, :] = Q_mat
        KK_mat[b, i, :, :] = K_mat
        VV_mat[b, i, :, :] = V_mat
        OO_mat[b, i, :, :] = O_mat


Q_mat = QQ_mat
K_mat = KK_mat
V_mat = VV_mat
O_mat = OO_mat
print("Q_mat", Q_mat.shape)
#print("Q_mat", Q_mat)
print("K_mat", K_mat.shape)
print("V_mat", V_mat.shape)
print("O_mat", O_mat.shape)
#exit(0)




Q_mat = Q_mat.to(torch.float)
K_mat = K_mat.to(torch.float)
V_mat = V_mat.to(torch.float)
O_mat = O_mat.to(torch.float)

#print("Q_mat", Q_mat)


#print("K_mat.transpose(-2, -1)", K_mat.transpose(-2, -1).shape)


expected_softmax = torch.softmax((Q_mat * (1/8)) @ K_mat.transpose(-2, -1), dim=-1)
print("expected_softmax", expected_softmax.shape)
#exit(0)


expected_attention = expected_softmax @ V_mat

#print("expected_attention", expected_attention)

ssum0 = torch.pow(expected_attention - O_mat, 2).sum()
ssum1 = torch.pow(O_mat, 2).sum()

ssum0 = torch.sqrt(ssum0)
ssum1 = torch.sqrt(ssum1)
print("relative error = ", ssum0/ssum1)
#exit(0)


#---------------------


import torch
import numpy as np
import os
from typing import Union, Optional

def save_bfloat16_tensor(tensor: torch.Tensor, filepath: str) -> bool:
    """
    Save a bfloat16/float16 tensor to a numpy file as uint16.
    This preserves the raw bits of the floating point values.

    Args:
        tensor: Input tensor (will be converted to bfloat16 if not already)
        filepath: Path to save the numpy file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Convert to bfloat16 if not already
        if tensor.dtype != torch.bfloat16:
            try:
                tensor = tensor.to(dtype=torch.bfloat16)
            except TypeError:
                print("Warning: bfloat16 not supported, falling back to float16")
                tensor = tensor.to(dtype=torch.float16)

        # Get the raw bytes and convert to uint16
        if tensor.dtype == torch.bfloat16:
            # For bfloat16, we need to handle the conversion carefully
            # Convert to float32 first to preserve precision
            float32_tensor = tensor.to(dtype=torch.float32)
            # Get the raw bytes and reshape to uint16
            uint16_array = float32_tensor.numpy().view(np.uint32) >> 16
        else:  # float16
            # For float16, we can directly view as uint16
            uint16_array = tensor.numpy().view(np.uint16)

        # Save to file
        np.save(filepath, uint16_array)
        print(f"Successfully saved tensor to {filepath} as uint16")
        print(f"Raw uint16 values: {uint16_array}")
        return True

    except Exception as e:
        print(f"Error saving tensor: {e}")
        return False

def load_bfloat16_tensor(filepath: str, device: Optional[Union[str, torch.device]] = None) -> Optional[torch.Tensor]:
    """
    Load a numpy uint16 file into a bfloat16/float16 tensor.
    The uint16 values are interpreted as the raw bits of bfloat16/float16.

    Args:
        filepath: Path to the numpy file
        device: Device to load the tensor onto (e.g., 'cpu', 'cuda:0')

    Returns:
        Optional[torch.Tensor]: Loaded tensor or None if loading failed
    """
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        # Load numpy array as uint16
        uint16_array = np.load(filepath)

        try:
            # Try bfloat16 first
            # Convert uint16 to float32 bits and then to bfloat16

            float32_bits = (uint16_array.astype(np.uint32) << 16)
            float32_array = float32_bits.view(np.float32)

            tensor = torch.from_numpy(float32_array).to(dtype=torch.bfloat16)
        except TypeError:
            print("Warning: bfloat16 not supported, falling back to float16")
            # For float16, directly view as float16
            tensor = torch.from_numpy(uint16_array.view(np.float16)).to(dtype=torch.float16)

        # Move to specified device if provided
        if device is not None:
            tensor = tensor.to(device)

        print(f"Successfully loaded tensor from {filepath}")
        return tensor

    except Exception as e:
        print(f"Error loading tensor: {e}")
        return None



Q1_mat = Q_mat.to(torch.bfloat16)
K1_mat = K_mat.to(torch.bfloat16)
V1_mat = V_mat.to(torch.bfloat16)
O1_mat = O_mat.to(torch.bfloat16)

save_bfloat16_tensor(Q1_mat, "13b_16h_q.npy")
save_bfloat16_tensor(K1_mat, "13b_16h_k.npy")
save_bfloat16_tensor(V1_mat, "13b_16h_v.npy")
save_bfloat16_tensor(O1_mat, "13b_16h_o.npy")


#------------------------




Q2_mat = load_bfloat16_tensor("13b_16h_q.npy")

K2_mat = load_bfloat16_tensor("13b_16h_k.npy")

V2_mat = load_bfloat16_tensor("13b_16h_v.npy")

O2_mat = load_bfloat16_tensor("13b_16h_o.npy")


Q_mat = Q_mat.to(torch.bfloat16)
K_mat = K_mat.to(torch.bfloat16)
V_mat = V_mat.to(torch.bfloat16)
O_mat = O_mat.to(torch.bfloat16)

print("Q_mat", Q_mat.shape)
print("Q2_mat", Q2_mat.shape)
assert torch.allclose(Q_mat, Q2_mat)
assert torch.allclose(K_mat, K2_mat)
assert torch.allclose(V_mat, V2_mat)
assert torch.allclose(O_mat, O2_mat)
