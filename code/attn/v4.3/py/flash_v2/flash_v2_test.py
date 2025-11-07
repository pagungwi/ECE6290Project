import torch


import numpy as np





N = 1024*1 + 1 # token length

d = 64 # d_model

Br = 4
Bc = 16


#Q_mat = torch.rand((N, d))
#K_mat = torch.rand((N, d))
#V_mat = torch.rand((N, d))


def np_uint16_to_bfloat16_tensor(arr_uint16: np.ndarray):
    assert arr_uint16.dtype == np.uint16
    arr_uint8 = arr_uint16.view(np.uint8)
    t = torch.from_numpy(arr_uint8).view(torch.int16)
    return t.view(torch.bfloat16)



q_uint16 = np.load("../npy_save/q_uint16.npy")
Q_mat = np_uint16_to_bfloat16_tensor(q_uint16)

k_uint16 = np.load("../npy_save/k_uint16.npy")
K_mat = np_uint16_to_bfloat16_tensor(k_uint16)

v_uint16 = np.load("../npy_save/v_uint16.npy")
V_mat = np_uint16_to_bfloat16_tensor(v_uint16)

o_uint16 = np.load("../npy_save/o_uint16.npy")
O_mat = np_uint16_to_bfloat16_tensor(o_uint16)



print("Q_mat", Q_mat.shape)
print(Q_mat)

print("K_mat", K_mat.shape)
print(K_mat)

print("V_mat", V_mat.shape)
print(V_mat)

print("O_mat", O_mat.shape)
print(O_mat)

#exit(0)


# Q_mat @ K_mat.T: shape = [N,N]
Q_mat = Q_mat * (1/8)
expected_softmax = torch.softmax(Q_mat @ K_mat.T, dim=1)

# shape = [N,d]
expected_attention = expected_softmax @ V_mat



#assert torch.allclose(O_mat, expected_attention)


print("\n calc_attention: \n")
print(expected_attention.shape)
print(expected_attention)
print("\n\n")


print("reference O_mat: \n")
print(O_mat.shape)
print(O_mat)
print("\n\n")


loss = torch.nn.functional.mse_loss(O_mat.view(torch.float),
                                    expected_attention.view(torch.float))
print("mse loss: ", loss)

loss2 = torch.abs(O_mat - expected_attention).sum()
print("sum loss2: ", loss2)


exit(0)

print("--------------------------------")





#---------------------------------------------------------------------

N1 = N - 1
N2 = 1


Q_rem = torch.zeros((4,  d), dtype=torch.bfloat16)
K_rem = torch.zeros((16, d), dtype=torch.bfloat16)
V_rem = torch.zeros((16, d), dtype=torch.bfloat16)


Q_rem[0, :] = Q_mat[N1, :]
K_rem[0, :] = K_mat[N1, :]
V_rem[0, :] = V_mat[N1, :]

# print("Q_rem", Q_rem.shape)
# print("K_rem", K_rem.shape)
# print("V_rem", V_rem.shape)
# exit(0)

assert torch.allclose(Q_rem[0,:], Q_mat[N1,:])
assert torch.allclose(K_rem[0,:], K_mat[N1,:])
assert torch.allclose(V_rem[0,:], V_mat[N1,:])

#print("Q_rem", Q_rem)
#print("K_rem", K_rem)
#print("V_rem", V_rem)
#exit(0)



O = torch.zeros((N, d), dtype=torch.bfloat16)



# Q: [N1=1024, 64]
for block_start_Br in range(0, N1, Br):
    block_end_Br = block_start_Br + Br

    Qi = Q_mat[block_start_Br:block_end_Br, :]  # shape Br x d

    Oi = torch.zeros((Br, d), dtype=torch.bfloat16)  # shape Br x d
    li = torch.zeros((Br, 1))  # shape Br x 1
    mi = torch.full((Br, 1), -torch.inf, dtype=torch.bfloat16)  # shape Br x 1


    # K,V: [N1=1024, 64]
    for block_start_Bc in range(0, N1, Bc):
        block_end_Bc = block_start_Bc + Bc

        Kj = K_mat[block_start_Bc:block_end_Bc, :]  # shape Bc x d
        Vj = V_mat[block_start_Bc:block_end_Bc, :]  # shape Bc x d

        Sij = Qi @ Kj.T


        #---------------------------------------------------
        # have a scaling factor: torch.exp(mij_hat - mi_new)
        #---------------------------------------------------
        '''
        mij_hat = torch.max(Sij, dim=1).values[:, None]

        pij_hat = torch.exp(Sij - mij_hat)

        lij_hat = torch.sum(pij_hat, dim=1)[:, None]

        mi_new = torch.max(torch.column_stack([mi, mij_hat]), dim=1).values[:, None]

        # only keep just one li
        #li_new = torch.exp(mi - mi_new) * li + torch.exp(mij_hat - mi_new) * lij_hat
        li = torch.exp(mi - mi_new) * li + torch.exp(mij_hat - mi_new) * lij_hat

        # no need to multiply li
        Oi = (torch.exp(mi - mi_new) * Oi) + (torch.exp(mij_hat - mi_new) * pij_hat) @ Vj

        mi = mi_new
        '''

        #---------------------------------------------------
        # no scaling factor: torch.exp(mij_hat - mi_new)
        #---------------------------------------------------
        #'''
        mij_hat = torch.max(Sij, dim=1).values[:, None]

        mi_new = torch.max(torch.column_stack([mi, mij_hat]), dim=1).values[:, None]

        pij_hat = torch.exp(Sij - mi_new)
        lij_hat = torch.sum(pij_hat, dim=1)[:, None]

        li = torch.exp(mi - mi_new) * li + lij_hat

        Oi = Oi * torch.exp(mi - mi_new) + pij_hat @ Vj

        mi = mi_new
        #'''
        #---------------------------------------------------


    #---------------------------------------------------
    # remaining K,V
    #---------------------------------------------------
    # K,V: [N2=1, 64]

    Sij = Qi @ K_rem.T # [4,16]
    #print("Sij:", Sij)

    # mij_hat.shape = [4,1]
    mij_hat = Sij[:, 0][:, None]
    #print("mij_hat:", mij_hat)

    # mi_new.shape = [4,1]
    mi_new = torch.max(torch.column_stack([mi, mij_hat]), dim=1).values[:, None]

    # [4,16]
    #pij_hat = Sij.clone()

    #pij_hat[:,0] = torch.exp(Sij[:,0][:, None] - mi_new).view(4*1)
    #print("pij_hat:", pij_hat)

    pij_hat = torch.exp(Sij - mi_new)


    # [4,1]
    lij_hat = pij_hat[:,0][:, None]
    #print("lij_hat:", lij_hat)


    # [4,1]
    correction = torch.exp(mi - mi_new)
    #print("correction:", correction) # could be all 1 most of the time.


    li = correction * li + lij_hat

    # pij_hat @ V_rem: using element-wise mul on NPU to speed up computation
    Oi = correction * Oi + pij_hat @ V_rem

    #---------------------------------------------------


    Oi = Oi / li
    O[block_start_Br:block_end_Br, :] = Oi





#---------------------------------------------------
# remaining Q
#---------------------------------------------------

Qi = Q_rem  # shape 4 x d

Oi = torch.zeros((4, d))  # shape Br x d
li = torch.zeros((4, 1))  # shape Br x 1
mi = torch.full((4, 1), -torch.inf)  # shape Br x 1


# K,V: [N1=1024, 64]
for block_start_Bc in range(0, N1, Bc):
    block_end_Bc = block_start_Bc + Bc

    Kj = K_mat[block_start_Bc:block_end_Bc, :]  # shape Bc x d
    Vj = V_mat[block_start_Bc:block_end_Bc, :]  # shape Bc x d

    Sij = Qi @ Kj.T


    #---------------------------------------------------
    # have a scaling factor: torch.exp(mij_hat - mi_new)
    #---------------------------------------------------
    '''
        mij_hat = torch.max(Sij, dim=1).values[:, None]

        pij_hat = torch.exp(Sij - mij_hat)

        lij_hat = torch.sum(pij_hat, dim=1)[:, None]

        mi_new = torch.max(torch.column_stack([mi, mij_hat]), dim=1).values[:, None]

        # only keep just one li
        #li_new = torch.exp(mi - mi_new) * li + torch.exp(mij_hat - mi_new) * lij_hat
        li = torch.exp(mi - mi_new) * li + torch.exp(mij_hat - mi_new) * lij_hat

        # no need to multiply li
        Oi = (torch.exp(mi - mi_new) * Oi) + (torch.exp(mij_hat - mi_new) * pij_hat) @ Vj

        mi = mi_new
    '''

    #---------------------------------------------------
    # no scaling factor: torch.exp(mij_hat - mi_new)
    #---------------------------------------------------
    #'''
    mij_hat = torch.max(Sij, dim=1).values[:, None]

    mi_new = torch.max(torch.column_stack([mi, mij_hat]), dim=1).values[:, None]

    pij_hat = torch.exp(Sij - mi_new)

    lij_hat = torch.sum(pij_hat, dim=1)[:, None]

    correction = torch.exp(mi - mi_new)

    li = correction * li + lij_hat

    Oi = correction * Oi + pij_hat @ Vj

    mi = mi_new
    #'''
    #---------------------------------------------------


#---------------------------------------------------
# remaining K,V
#---------------------------------------------------
# K,V: [N2=1, 64]

Sij = Qi @ K_rem.T # [4,16]

# [4,1]
mij_hat = torch.max(Sij, dim=1).values[:, None]

# [4,1]
mi_new = torch.max(torch.column_stack([mi, mij_hat]), dim=1).values[:, None]

# [4,16]
pij_hat = Sij.clone()

pij_hat[0,0] = torch.exp(Sij[0,0] - mi_new[0,0]).view(1*1)


# [4,1]
lij_hat = pij_hat[:,0][:, None]

# [4,1]
correction = torch.exp(mi - mi_new)

li = correction * li + lij_hat

Oi = correction * Oi + pij_hat @ V_rem

mi = mi_new

#---------------------------------------------------



Oi = Oi / li
O[N1, :] = Oi[0,:]

#---------------------------------------------------



#assert torch.allclose(O[:1024, :], expected_attention[:1024, :])
#assert torch.allclose(O, expected_attention)


print("expected_attention: \n")
print(O_mat.shape)
print(O_mat)
print("\n\n")


print("O: \n")
print(O.shape)
print(O)
print("\n\n")


loss = torch.nn.functional.mse_loss(O_mat.view(torch.float),
                                    O.view(torch.float))
print("mse loss: ", loss)
