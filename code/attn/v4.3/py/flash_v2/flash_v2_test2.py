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



q_uint16 = np.load("../npy_save/q_uint16.npy")
Q_mat = np_uint16_to_bfloat16_tensor(q_uint16)

k_uint16 = np.load("../npy_save/k_uint16.npy")
K_mat = np_uint16_to_bfloat16_tensor(k_uint16)

v_uint16 = np.load("../npy_save/v_uint16.npy")
V_mat = np_uint16_to_bfloat16_tensor(v_uint16)

o_uint16 = np.load("../npy_save/o_uint16.npy")
O_mat = np_uint16_to_bfloat16_tensor(o_uint16)



#print("Q_mat", Q_mat.shape)
#print(Q_mat[(0*64):(0+1)*64,:])
#print(Q_mat)
#exit(0)

#print("K_mat", K_mat.shape)
#print(K_mat)
#print(K_mat[((3*64)):((3+1)*64),:])
#exit(0)



#print("V_mat", V_mat.shape)
#print(V_mat)
#print(V_mat[((1*64)):((1+1)*64),:])
#exit(0)


# print("O_mat", O_mat.shape)
# print(O_mat)

# exit(0)


#Q_mat @ K_mat.T: shape = [N,N]

# Q_mat = Q_mat * (1/8)
# expected_softmax = torch.softmax(Q_mat @ K_mat.T, dim=1)

# # shape = [N,d]
# expected_attention = expected_softmax @ V_mat



# ssum0 = torch.pow(expected_attention - O_mat, 2).sum()
# ssum1 = torch.pow(O_mat, 2).sum()

# ssum0 = torch.sqrt(ssum0)
# ssum1 = torch.sqrt(ssum1)
# print("relative error = ", ssum0/ssum1) # 24%

# exit(0)



# #assert torch.allclose(O_mat, expected_attention)


# print("\n calc_attention: \n")
# print(expected_attention.shape)
# print(expected_attention)
# # print("\n\n")


# # print("reference O_mat: \n")
# print(O_mat.shape)
# print(O_mat)
# # # print("\n\n")


# ssum0 = torch.pow(expected_attention - O_mat, 2).sum()
# ssum1 = torch.pow(O_mat, 2).sum()

# ssum0 = torch.sqrt(ssum0)
# ssum1 = torch.sqrt(ssum1)
# print("relative error = ", ssum0/ssum1) # 24%

# exit(0)




# print("--------------------------------")
# print("--------------------------------")





#---------------------------------------------------------------------

Q_mat = Q_mat.to(torch.float)
K_mat = K_mat.to(torch.float)
V_mat = V_mat.to(torch.float)
O_mat = O_mat.to(torch.float)


# print("Q_mat", Q_mat[0:64,:].shape)
# print("Q_mat", Q_mat[0:64,:])
# exit(0)




Q_mat = Q_mat * 0.125


N = 1024

O = torch.zeros((N, d))



#for block_start_Br in range(0, N, Br):
for block_start_Br in range(0, 64, Br):
    block_end_Br = block_start_Br + Br

    Qi = Q_mat[block_start_Br:block_end_Br, :]  # shape Br x d

    Oi = torch.zeros((Br, d))  # shape Br x d

    li = torch.zeros((Br, 1))  # shape Br x 1
    mi = torch.full((Br, 1), -torch.inf)  # shape Br x 1

    #for block_start_Bc in range(0, N, Bc):
    for block_start_Bc in range(0, 2*64, Bc):
    #for block_start_Bc in range(0, 2*16, Bc):
    #for block_start_Bc in range(1):
        block_end_Bc = block_start_Bc + Bc

        Kj = K_mat[block_start_Bc:block_end_Bc, :]  # shape Bc x d
        Vj = V_mat[block_start_Bc:block_end_Bc, :]  # shape Bc x d

        # print("Qi.dtype:", Qi.dtype)
        # print("Kj.dtype:", Kj.dtype)

        Sij = Qi @ Kj.T

        # print("Sij:", Sij.shape)
        # print("Sij:", Sij)


        # Oi_float = Sij.cpu().numpy()
        # print("save ....")
        # np.save("oi_float.npy", Oi_float)
        # #exit(0)



        #'''
        mij_hat = torch.max(Sij, dim=1).values[:, None]

        mi_new = torch.max(torch.column_stack([mi, mij_hat]), dim=1).values[:, None]

        # print("mi", mi)
        #print("mi_new", mi_new)
        #exit(0)


        # print("Sij - mi_new:", Sij - mi_new)
        # Oi = Sij - mi_new
        # Oi_float = Oi.view(torch.float).cpu().numpy()
        # print("save ....")
        # np.save("oi_float.npy", Oi_float)
        # exit(0)


        pij_hat = torch.exp(Sij - mi_new)
        #pij_hat = Sij - mi_new


        # print("pij_hat", pij_hat.shape)
        # print("pij_hat", pij_hat)

        # #Oi = pij_hat
        # Oi = pij_hat[:,8:16]
        # Oi_float = Oi.cpu().numpy()
        # print("save ....")
        # np.save("oi_float.npy", Oi_float)
        # # exit(0)



        lij_hat = torch.sum(pij_hat, dim=1)[:, None]
        # print("lij_hat", lij_hat.shape)
        # print("lij_hat", lij_hat)

        # Oi_float = lij_hat.cpu().numpy()
        # print("save ....")
        # np.save("oi_float.npy", Oi_float)
        # exit(0)




        correction = torch.exp(mi - mi_new)

        # print("mi - mi_new:", mi - mi_new)
        # print("correction", correction)
        #exit(0)


        li = correction * li + lij_hat
        #print("li:", li)
        # exit(0)


        # pij_hat[:,1:] = torch.zeros((15,))
        # kkk = pij_hat @ Vj
        # print(kkk.shape)
        # print(kkk)
        # exit(0)


        # print("Vj:", Vj.shape)
        # print("Vj:", Vj)

        # Oi_uint16 = Vj.cpu().numpy()
        # print("save ....")
        # np.save("oi_float.npy", Oi_uint16)
        # exit(0)



        Oi = correction * Oi + pij_hat @ Vj
        #Oi = Oi + pij_hat @ Vj
        #Oi = correction * Oi
        #Oi = pij_hat @ Vj

        # print("Oi:", Oi.shape)
        # print("Oi:", Oi)


        # # # #KKK = Oi[:,(4*8):(4+1)*8]

        # # # #Oi_uint16 = KKK.cpu().numpy()
        # Oi_uint16 = Oi.cpu().numpy()
        # print("save ....")
        # np.save("oi_float.npy", Oi_uint16)
        # #exit(0)






        mi = mi_new
        #print("mi:", mi)

        #print("break ...")
        #break
        #'''
        #---------------------------------------------------



    #exit(0)


    Oi = Oi / li
    #Oi = Oi
    O[block_start_Br:block_end_Br, :] = Oi

    #print("li :", li.shape)

    #print("li :", li)
    # exit(0)


    # print("Oi :", Oi.shape)
    # print("Oi :", Oi)
    # #print("Oi :", Oi[:,13*4:(13+1)*4])

    # PK = torch.tensor([], dtype=torch.float32)

    # for i in range(16):
    #     a = Oi[:,i*4:(i+1)*4].reshape(16)
    #     PK = torch.cat((PK, a))

    # Oi_float = PK.cpu().numpy()
    # print("Oi_float", Oi_float)
    # print("save ....")
    # np.save("oi_float.npy", Oi_float)
    # exit(0)


    # print("Oi :", Oi.shape)
    # print("Oi :", Oi)


    # Oi_float = Oi.cpu().numpy()
    # print("save ....")
    # np.save("oi_float.npy", Oi_float)

    # exit(0)


Oi = O[0:64,:]
Oi_float = Oi.cpu().numpy()
print("save ....")
np.save("oi_float.npy", Oi_float)

exit(0)




#print("O_mat:", O_mat[:1024, :])
#print("----------------------")

O = O[0:4,:]
ref = O_mat[0:4,:]

print("O:", O)
print("ref:", ref)


ssum0 = torch.pow(O - ref, 2).sum()
ssum1 = torch.pow(ref, 2).sum()

ssum0 = torch.sqrt(ssum0)
ssum1 = torch.sqrt(ssum1)

print("relative error = ", ssum0/ssum1) # float mode, error = 0.1%


#assert torch.allclose(O[:1024, :], O_mat[:1024, :])
