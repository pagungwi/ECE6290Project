//===- zero.cc --------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef ZERO_CC
#define ZERO_CC

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>


// need to know n at compile time
// in order to unroll the for-loop
template <typename T, int N>
//inline void zero_vectorized(T *__restrict c) {
 void zero_vectorized(T *__restrict c) {

    constexpr int r = 512 / (sizeof(T) * 8); // 512 bit store units for AIE2P
    static_assert(N % r == 0);

    const aie::vector<T, r> zeros = aie::zeros<T, r>();
    const T *__restrict c_end = c + N;


    //event0();

    for (; c < c_end; c += r)
      chess_prepare_for_pipelining chess_loop_range(r,)
      {
          aie::store_v(c, zeros);
      }

    //event1();
}





static inline bfloat16 get_neg_inf() {
    uint16_t raw = 0xFF80;
    return *reinterpret_cast<bfloat16*>(&raw);
}

// T = bfloat16
template <typename T, unsigned N> // N is defined in compile time
void neg_inf_vectorized(T *__restrict c)
{
  // sizeof(T): the bytes of the T type
  constexpr int r = 512 / (sizeof(T) * 8); // one 512 bit store unit

  // ensure n % r = 0

  //const aie::vector<T, r> zeros = aie::zeros<T, r>();

  // because the lut-exp cannot handle this, we cannot use this
  const bfloat16 abc = get_neg_inf();
  const aie::vector<T, r> value = aie::broadcast<T, r>((T)abc);

  // lut-exp: lower-bound=-128, e^(-128)=0
  //const aie::vector<T, r> value = aie::broadcast<T, r>((T)-65536000000.0f);

  const T *__restrict c_end = c + N; // end addr

  //event0();

  for (; c < c_end; c += r)
      chess_prepare_for_pipelining chess_loop_range(r,)
      {
          // later use chess_pipeline to unroll for optimization
          aie::store_v(c, value);
      }

  //event1();
}



extern "C" {

    void kernel_func_zero_bf16_o(bfloat16 *data)
    {
        // o: 64x64
        zero_vectorized<bfloat16, 64*64>(data);
    }

    void kernel_func_zero_float_o(float *data)
    {
        // o: 64x64
        zero_vectorized<float, 64*64>(data);
    }

    void kernel_func_neg_inf_bf16_m(bfloat16 *data)
    {
        neg_inf_vectorized<bfloat16, 64/4*16>(data);
    }

    void kernel_func_zero_float_l(float *data)
    {
        zero_vectorized<float, 64/4*8>(data);
    }

}

#endif
