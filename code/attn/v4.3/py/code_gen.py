for i in range(64):
    print(f"    aie::vector<bfloat16, 64> P{i} = aie::sub(S{i}, (bfloat16)mi_new[{i}]);")

exit(0)





for i in range(64):
    print(f"    aie::vector<bfloat16, 64> S{i}; // row {i}")

exit(0)




for i in range(64):
    print(f"    row_{i}.store(out + {i * 64});")

exit(0)


for i in range(64):
    print(f"                                              aie::vector<T_out, 64> row_{i};")





exit(0)



for i in range(8,16):

    print(f"""
          row_{i} = aie::concat(vec_10.template extract<8>({i-8}),
                              vec_11.template extract<8>({i-8}),
                              vec_12.template extract<8>({i-8}),
                              vec_13.template extract<8>({i-8}),
                              vec_14.template extract<8>({i-8}),
                              vec_15.template extract<8>({i-8}),
                              vec_16.template extract<8>({i-8}),
                              vec_17.template extract<8>({i-8}),
                              );
    """)



exit(0)

for i in range(64):
    print(f"                                              aie::vector<T_out, 64> row_{i},")





exit(0)


for i in range(15):
    print(f"P5[{i+1}] = 0;")


for i in range(15):
    print(f"P6[{i+1}] = 0;")

for i in range(15):
    print(f"P7[{i+1}] = 0;")
