import torch
import numpy as np


#----------------------------------------
# each tile
q_dim0 = 4
q_dim1 = 64

k_dim0 = 16 * 8
k_dim1 = 64

v_dim0 = 16 * 8
v_dim1 = 64

o_dim0 = 4
o_dim1 = 64
#----------------------------------------


#----------------------------------------
# Global Input
q_g_dim0 = 4
q_g_dim1 = 64

k_g_dim0 = 16 * 8
k_g_dim1 = 64

v_g_dim0 = 16 * 8
v_g_dim1 = 64

o_g_dim0 = 4
o_g_dim1 = 64

#q_g_ty  = np.ndarray[(q_g_dim0, q_g_dim1), np.dtype[bfloat16]]
#k_g_ty  = np.ndarray[(k_g_dim0, k_g_dim1), np.dtype[bfloat16]]
#v_g_ty  = np.ndarray[(v_g_dim0, v_g_dim1), np.dtype[bfloat16]]
#o_g_ty  = np.ndarray[(o_g_dim0, o_g_dim1), np.dtype[bfloat16]]

q_sz = q_g_dim0 * q_g_dim1
k_sz = k_g_dim0 * k_g_dim1
v_sz = v_g_dim0 * v_g_dim1
o_sz = o_g_dim0 * o_g_dim1
#-------------------------------------------------------------



# range: [0,1]
q = torch.rand(q_dim0, q_dim1, dtype=torch.bfloat16)
k = torch.rand(k_dim0, k_dim1, dtype=torch.bfloat16)
v = torch.rand(v_dim0, v_dim1, dtype=torch.bfloat16)
o = torch.rand(o_dim0, o_dim1, dtype=torch.bfloat16)


q_np = q.to(dtype=torch.float32).numpy()
np.save("q_np_f32.npy", q_np)

k_np = k.to(dtype=torch.float32).numpy()
np.save("k_np_f32.npy", k_np)

v_np = v.to(dtype=torch.float32).numpy()
np.save("v_np_f32.npy", v_np)

v_np = v.to(dtype=torch.float32).numpy()
np.save("v_np_f32.npy", v_np)


#q_bf16 = torch.from_numpy(q_np).to(torch.bfloat16)
