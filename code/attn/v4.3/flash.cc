//===- scale.cc -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define __AIENGINE__ 2
#define NOCPP
#define __AIEARCH__ 20




#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>


//#include "optimization_pragmas.h"


#include <aie_api/aie.hpp>

#include "zero.cc"







//----------------------------------------------------------


#if 0

#include <lut_based_ops.h>
#include <lut_based_ops.cpp>


static aie::vector<bfloat16, 16> exp_bf16_func_ori(aie::vector<bfloat16, 16> in_vec)
{
    v16bfloat16 vec_in = in_vec;
    v16accfloat acc_exp = getExpBf16(vec_in);

    v16bfloat16 bf16_exp = to_v16bfloat16(acc_exp);

    return bf16_exp;
}


aie::vector<bfloat16, 16> exp_simple_bf16(aie::vector<bfloat16, 16>  in_vec)
{
    aie::vector<float, 16> log2_e = broadcast_to_v16float(1.442695040888963f);
    aie::vector<float, 16> ln2 = broadcast_to_v16float(0.6931471805599453f);
    aie::vector<float, 16> scale = broadcast_to_v16float(0.2401598148889220f);
    aie::vector<float, 16> scale_1 = broadcast_to_v16float(1.0f);


    v16bfloat16 in = in_vec;


    aie::vector<float, 16> x = v16float(ups_to_v16accfloat(in));

    aie::accum<accfloat, 16> acc_prod = aie::mul(x, log2_e);


    //v16int32 ix = static_cast<v16int32>(prod); // still maintain fractions

    //v16accfloat ix_f = v16accfloat(ix);
    //v16accfloat _fx = prod - ix_f;

    aie::vector<float, 16> prod = acc_prod.to_vector<float>(); // accum to vector

    aie::vector<int32, 16> ix = aie::to_fixed<int32>(prod,0); // remove fractions

    aie::vector<float, 16> ix_f = aie::to_float(ix, 0);// fix to float

    aie::vector<float, 16> fx = aie::sub(prod, ix_f);


    //out = to_v16bfloat16(ix_f);
    //out = to_v16bfloat16(fx);
    //return;
    // above good


    /*
    aie::vector<float, 16> bbb = v16float(prod);

    aie::vector<int32, 16> truncated = aie::to_fixed<int32>(bbb,0);

    aie::vector<float, 16> kkk = aie::to_float(truncated, 0);

    v16accfloat cccc = v16accfloat(kkk);

    // removed fraction successful
    // on testing
    out = to_v16bfloat16(cccc);

    return;
*/


    aie::vector<int32, 16> int_127 = aie::broadcast<int32, 16>(127);

    ix = aie::add(ix, int_127);

    ix = aie::upshift(ix, 23); // ix << 23

    //void* aaa = static_cast<void*>ix;


    //aie::vector<float, 16> pow2_ix = aie::to_float(ix, 23);// fix to float
    //aie::vector<float, 16> pow2_ix = aie::load_v<16>((float*)aaa);
    //aie::vector<float, 16> pow2_ix = reinterpret_cast<float>((float*)aaa);

    // reinterpret_cast: not static_cast (this keeps number convert)
    aie::vector<float, 16> pow2_ix = aie::vector_cast<float>(ix);

    //out = to_v16bfloat16(pow2_ix);
    //return;
    // above good




    aie::accum<accfloat, 16> _p1 = aie::mul(ln2, fx);
    aie::vector<float, 16> p1 = v16float(_p1);

    aie::accum<accfloat, 16> _p2 = aie::mul(fx, fx);
    aie::vector<float, 16> p2 = v16float(_p2);

    aie::accum<accfloat, 16> _p3 = aie::mul(p2, scale);
    aie::vector<float, 16> p3 = v16float(_p3);


    p3 = aie::add(p3, scale_1);
    p3 = aie::add(p3, p1);

    aie::vector<float, 16> pow2_fx = p3;

    aie::accum<accfloat, 16> res = aie::mul(pow2_ix , pow2_fx);


    v16bfloat16 out = to_v16bfloat16(res);

    return out;
}


#endif




// Calculate the e^(x) function as 2^(log2e * x)
static inline aie::vector<bfloat16,16> exp_bf16_func(aie::vector<bfloat16,16> in)
{
    const float log2e = 1.44269504089f;

    aie::vector<bfloat16,16> log2e_vec = aie::broadcast<bfloat16,16>(log2e);

    aie::accum<accfloat, 16> exp_in = aie::mul(in, log2e_vec);

    aie::vector<bfloat16, 16> exp_val = aie::exp2<bfloat16>(exp_in.to_vector<float>());

    return exp_val;
}










//--------------------------------------------------------------------




//--------------------------------------------------------------------
#if 1

template <typename T_in, typename T_out,
          unsigned rowA, unsigned colA, unsigned colB,
          unsigned r, unsigned s, unsigned t>
static inline void matmul_m8k64n16_vectorized_1x2(const T_in *__restrict A,
                                           const T_in *__restrict B,

                                           // 8x16
                                           aie::vector<T_out, 16>& row0,
                                           aie::vector<T_out, 16>& row1,
                                           aie::vector<T_out, 16>& row2,
                                           aie::vector<T_out, 16>& row3,
                                           aie::vector<T_out, 16>& row4,
                                           aie::vector<T_out, 16>& row5,
                                           aie::vector<T_out, 16>& row6,
                                           aie::vector<T_out, 16>& row7
                                           )
{
    using MMUL = aie::mmul<r, s, t, T_in, T_in, acc32>;


    const T_in *__restrict pA0 = A;

    const T_in *__restrict pB0 = B;
    const T_in *__restrict pB1 = B + 8 * MMUL::size_B;
    //---------------------------------------------------


    /*
     * transpose: done by previous mmul projection operation.
     */

    // remove all transpose there: reduce 42 cycles in cost

    //event0();

    aie::vector<T_in, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA0);
    pA0 += MMUL::size_A;

    aie::vector<T_in, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB0);
    B0 = aie::transpose(B0, r, s);
    pB0 += MMUL::size_B;

    aie::vector<T_in, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB1);
    B1 = aie::transpose(B1, r, s);
    pB1 += MMUL::size_B;


    MMUL C00;
    MMUL C01;

    C00.mac(A0, B0);
    C01.mac(A0, B1);

    // o_C0 = C0.template to_vector<T_out>();
    // o_C1 = C1.template to_vector<T_out>();
    // o_C2 = C2.template to_vector<T_out>();
    // o_C3 = C3.template to_vector<T_out>();
    // return;


    //aie::vector<T_out, 16> abc = A0.template extract<16>(1);
    //o_C0 = abc;
    //return;

    //aie::vector<T_out, 16> abc = B0.template extract<16>(1);
    //o_C0 = abc;
    //return;


    for (unsigned i = 1; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
        chess_flatten_loop
        //chess_prepare_for_pipelining chess_loop_range(colA-1,)
#endif
        {
            A0 = aie::load_v<MMUL::size_A>(pA0);
            pA0 += MMUL::size_A;

            B0 = aie::load_v<MMUL::size_B>(pB0);
            B0 = aie::transpose(B0, r, s);
            pB0 += MMUL::size_B;

            B1 = aie::load_v<MMUL::size_B>(pB1);
            B1 = aie::transpose(B1, r, s);
            pB1 += MMUL::size_B;

            C00.mac(A0, B0);
            C01.mac(A0, B1);
        }


    //event1();



    //----------------------------------------------------------------
    // only take 1 cycle

    //event0();

    // MMUL::size_C: 8x8
    aie::vector<T_out, MMUL::size_C> t_C00 = C00.template to_vector<T_out>();
    aie::vector<T_out, MMUL::size_C> t_C01 = C01.template to_vector<T_out>();

    {
        // work on NPU2:
        // extract at least 8 elements: at least 128bit for each operation
        row0 = aie::concat(t_C00.template extract<8>(0),
                           t_C01.template extract<8>(0));

        row1 = aie::concat(t_C00.template extract<8>(1),
                           t_C01.template extract<8>(1));

        row2 = aie::concat(t_C00.template extract<8>(2),
                           t_C01.template extract<8>(2));

        row3 = aie::concat(t_C00.template extract<8>(3),
                           t_C01.template extract<8>(3));

        row4 = aie::concat(t_C00.template extract<8>(4),
                           t_C01.template extract<8>(4));

        row5 = aie::concat(t_C00.template extract<8>(5),
                           t_C01.template extract<8>(5));

        row6 = aie::concat(t_C00.template extract<8>(6),
                           t_C01.template extract<8>(6));

        row7 = aie::concat(t_C00.template extract<8>(7),
                           t_C01.template extract<8>(7));
    }

    //event1();
    //----------------------------------------------------------------

}


static inline void matmul_m8k64n16_vectorized_8x8x8_bf16_bf16(const bfloat16 *__restrict A,
                                                       const bfloat16 *__restrict B,

                                                       aie::vector<bfloat16, 16>& row0,
                                                       aie::vector<bfloat16, 16>& row1,
                                                       aie::vector<bfloat16, 16>& row2,
                                                       aie::vector<bfloat16, 16>& row3,
                                                       aie::vector<bfloat16, 16>& row4,
                                                       aie::vector<bfloat16, 16>& row5,
                                                       aie::vector<bfloat16, 16>& row6,
                                                       aie::vector<bfloat16, 16>& row7
                                                       )
{
  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;

  // [8,16] @ [64,16] = [8,16]
  constexpr int m = 8;
  constexpr int k = 64;
  constexpr int n = 16;

  matmul_m8k64n16_vectorized_1x2<bfloat16, bfloat16,
                                 (m / r), (k / s), (n / t), r, s, t>
                                 (A, B,
                                  row0, row1, row2, row3,
                                  row4, row5, row6, row7);
}




template <typename T_in, typename T_out,
          unsigned rowA, unsigned colA, unsigned colB,
          unsigned r, unsigned s, unsigned t>
static inline void matmul_m8k16n64_vectorized_1x4(const aie::vector<T_in, r*s>   A0,  // 8x8
                                           const aie::vector<T_in, r*s>   A1,  // 8x8
                                           const T_in *__restrict         B,  // [16,32]

                                           /*
                                             P_dot_V = [C00, C01, C02, C03]
                                           */
                                           aie::vector<T_out, r*t>&  C00,
                                           aie::vector<T_out, r*t>&  C01,
                                           aie::vector<T_out, r*t>&  C02,
                                           aie::vector<T_out, r*t>&  C03
                                           )
{
  using MMUL = aie::mmul<r, s, t, T_in, T_in, acc32>;


  const T_in *__restrict pB0 = B;
  const T_in *__restrict pB1 = B + colB * MMUL::size_B;



  aie::vector<T_in, MMUL::size_B> B0, B1;



  /*
   * total 1024bit registers: 4
   * 8x8x16bit = 1024bit
   * max C blocks = 2x2 = 4,
   * when higher than 4, errors appear.
   */

  { // C00

      // careful with this stuff: no initial value given, cause large error
      MMUL mac;

      B0 = aie::load_v<MMUL::size_B>(pB0);
      pB0 += MMUL::size_B;

      B1 = aie::load_v<MMUL::size_B>(pB1);
      pB1 += MMUL::size_B;

      mac.mac(A0, B0);
      mac.mac(A1, B1);

      C00 = mac.template to_vector<T_out>();
  }


  { // C01

      MMUL mac;

      B0 = aie::load_v<MMUL::size_B>(pB0);
      pB0 += MMUL::size_B;

      B1 = aie::load_v<MMUL::size_B>(pB1);
      pB1 += MMUL::size_B;

      mac.mac(A0, B0);
      mac.mac(A1, B1);

      C01 = mac.template to_vector<T_out>();
  }


  { // C02
      MMUL mac;

      B0 = aie::load_v<MMUL::size_B>(pB0);
      pB0 += MMUL::size_B;

      B1 = aie::load_v<MMUL::size_B>(pB1);
      pB1 += MMUL::size_B;

      mac.mac(A0, B0);
      mac.mac(A1, B1);

      C02 = mac.template to_vector<T_out>();
  }


  { // C03
      MMUL mac;

      B0 = aie::load_v<MMUL::size_B>(pB0);
      pB0 += MMUL::size_B;

      B1 = aie::load_v<MMUL::size_B>(pB1);
      pB1 += MMUL::size_B;

      mac.mac(A0, B0);
      mac.mac(A1, B1);

      C03 = mac.template to_vector<T_out>();
  }

}


static inline void reshape_p_dot_v(const aie::vector<bfloat16, 8*8> P_dot_v0,
                            const aie::vector<bfloat16, 8*8> P_dot_v1,
                            const aie::vector<bfloat16, 8*8> P_dot_v2,
                            const aie::vector<bfloat16, 8*8> P_dot_v3,
                            const aie::vector<bfloat16, 8*8> P_dot_v4,
                            const aie::vector<bfloat16, 8*8> P_dot_v5,
                            const aie::vector<bfloat16, 8*8> P_dot_v6,
                            const aie::vector<bfloat16, 8*8> P_dot_v7,

                            aie::vector<bfloat16, 64>& row0,
                            aie::vector<bfloat16, 64>& row1,
                            aie::vector<bfloat16, 64>& row2,
                            aie::vector<bfloat16, 64>& row3,
                            aie::vector<bfloat16, 64>& row4,
                            aie::vector<bfloat16, 64>& row5,
                            aie::vector<bfloat16, 64>& row6,
                            aie::vector<bfloat16, 64>& row7
                            )
{

    aie::vector<bfloat16, 8> c0, c1, c2, c3, c4, c5, c6, c7;


    { // row0
        c0 = P_dot_v0.template extract<8>(0);
        c1 = P_dot_v1.template extract<8>(0);
        c2 = P_dot_v2.template extract<8>(0);
        c3 = P_dot_v3.template extract<8>(0);
        c4 = P_dot_v4.template extract<8>(0);
        c5 = P_dot_v5.template extract<8>(0);
        c6 = P_dot_v6.template extract<8>(0);
        c7 = P_dot_v7.template extract<8>(0);

        row0 = aie::concat(c0, c1, c2, c3, c4, c5, c6, c7);
    }


    { // row1
        c0 = P_dot_v0.template extract<8>(1);
        c1 = P_dot_v1.template extract<8>(1);
        c2 = P_dot_v2.template extract<8>(1);
        c3 = P_dot_v3.template extract<8>(1);
        c4 = P_dot_v4.template extract<8>(1);
        c5 = P_dot_v5.template extract<8>(1);
        c6 = P_dot_v6.template extract<8>(1);
        c7 = P_dot_v7.template extract<8>(1);

        row1 = aie::concat(c0, c1, c2, c3, c4, c5, c6, c7);
    }


    { // row2
        c0 = P_dot_v0.template extract<8>(2);
        c1 = P_dot_v1.template extract<8>(2);
        c2 = P_dot_v2.template extract<8>(2);
        c3 = P_dot_v3.template extract<8>(2);
        c4 = P_dot_v4.template extract<8>(2);
        c5 = P_dot_v5.template extract<8>(2);
        c6 = P_dot_v6.template extract<8>(2);
        c7 = P_dot_v7.template extract<8>(2);

        row2 = aie::concat(c0, c1, c2, c3, c4, c5, c6, c7);
    }

    { // row3
        c0 = P_dot_v0.template extract<8>(3);
        c1 = P_dot_v1.template extract<8>(3);
        c2 = P_dot_v2.template extract<8>(3);
        c3 = P_dot_v3.template extract<8>(3);
        c4 = P_dot_v4.template extract<8>(3);
        c5 = P_dot_v5.template extract<8>(3);
        c6 = P_dot_v6.template extract<8>(3);
        c7 = P_dot_v7.template extract<8>(3);

        row3 = aie::concat(c0, c1, c2, c3, c4, c5, c6, c7);
    }

    { // row4
        c0 = P_dot_v0.template extract<8>(4);
        c1 = P_dot_v1.template extract<8>(4);
        c2 = P_dot_v2.template extract<8>(4);
        c3 = P_dot_v3.template extract<8>(4);
        c4 = P_dot_v4.template extract<8>(4);
        c5 = P_dot_v5.template extract<8>(4);
        c6 = P_dot_v6.template extract<8>(4);
        c7 = P_dot_v7.template extract<8>(4);

        row4 = aie::concat(c0, c1, c2, c3, c4, c5, c6, c7);
    }

    { // row5
        c0 = P_dot_v0.template extract<8>(5);
        c1 = P_dot_v1.template extract<8>(5);
        c2 = P_dot_v2.template extract<8>(5);
        c3 = P_dot_v3.template extract<8>(5);
        c4 = P_dot_v4.template extract<8>(5);
        c5 = P_dot_v5.template extract<8>(5);
        c6 = P_dot_v6.template extract<8>(5);
        c7 = P_dot_v7.template extract<8>(5);

        row5 = aie::concat(c0, c1, c2, c3, c4, c5, c6, c7);
    }

    { // row6
        c0 = P_dot_v0.template extract<8>(6);
        c1 = P_dot_v1.template extract<8>(6);
        c2 = P_dot_v2.template extract<8>(6);
        c3 = P_dot_v3.template extract<8>(6);
        c4 = P_dot_v4.template extract<8>(6);
        c5 = P_dot_v5.template extract<8>(6);
        c6 = P_dot_v6.template extract<8>(6);
        c7 = P_dot_v7.template extract<8>(6);

        row6 = aie::concat(c0, c1, c2, c3, c4, c5, c6, c7);
    }

    { // row7
        c0 = P_dot_v0.template extract<8>(7);
        c1 = P_dot_v1.template extract<8>(7);
        c2 = P_dot_v2.template extract<8>(7);
        c3 = P_dot_v3.template extract<8>(7);
        c4 = P_dot_v4.template extract<8>(7);
        c5 = P_dot_v5.template extract<8>(7);
        c6 = P_dot_v6.template extract<8>(7);
        c7 = P_dot_v7.template extract<8>(7);

        row7 = aie::concat(c0, c1, c2, c3, c4, c5, c6, c7);
    }

}



static inline void matmul_m8k16n64_vectorized_8x8x8_bf16_bf16(const aie::vector<bfloat16, 8*8> A0,
                                                       const aie::vector<bfloat16, 8*8> A1,

                                                       const bfloat16 *__restrict       B,

                                                       aie::vector<bfloat16, 64>& row0,
                                                       aie::vector<bfloat16, 64>& row1,
                                                       aie::vector<bfloat16, 64>& row2,
                                                       aie::vector<bfloat16, 64>& row3
                                                       )
{
  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;

  // [8,16] @ [16,64] = [8,64]
  constexpr int m = 8;
  constexpr int k = 16;
  constexpr int n = 64;


  matmul_m8k16n64_vectorized_1x4<bfloat16, bfloat16,
                                 (m / r), (k / s), (n / t), r, s, t>
                                 (A0, A1, B,
                                  row0, row1, row2, row3);

  // matmul_m8k16n64_vectorized_1x4<bfloat16, bfloat16,
  //                                (m / r), (k / s), (n / t), r, s, t>
  //                                (A0, A1, B + 4 * 8 * 8,
  //                                 row0, row1, row2, row3);


  // reshape_p_dot_v(row0, row1, row2, row3, row4, row5, row6, row7,
  //                 row0, row1, row2, row3, row4, row5, row6, row7);

}






/*
  MMUL.mac: on peno, results are wrong
  we need to use xchess here, because acc initial values are not from DM
 */

static inline void do_kernel_flash_v2_row4_col64(
                                   /*
                                    * q @ kt, s @ v : rst=484 mode

                                    * q: 4x8
                                    * kt: 8x4, each tile is transposed in kernel
                                    * v: 8x4
                                    */

                                   const bfloat16 *__restrict q, // [8,64]
                                   const bfloat16 *__restrict k, // [16,64]
                                   const bfloat16 *__restrict v, // [16,64]
                                   bfloat16       *__restrict y, // [8,64]

                                   // only 1st 8 are used
                                   aie::vector<float, 8>&        li,
                                   aie::vector<bfloat16, 16>&    mi,

                                   const bfloat16 q_scale_factor,


                                   // single row valid: k,v padding flag
                                   const int kv_padding_flag
                                   )

{
    //event0();

    /*

      Sij = q * k : [8,64]x[64,16]=[8,16]

      mij_hat = torch.max(Sij, dim=1).values[:, None] : [8,1]

      mi_new = torch.max(torch.column_stack([mi, mij_hat]), dim=1).values[:, None] : [8,1]

      pij_hat = torch.exp(Sij - mi_new) : [8,16]

      lij_hat = torch.sum(pij_hat, dim=1)[:, None] : [8,1]

      correction = torch.exp(mi - mi_new) : [8,1]

      Oi = correction * Oi + pij_hat @ Vj : [8,64]

      li = correction * li + lij_hat : [8,1]

      mi = mi_new
    */


    // q @ kt : [8x16]
    aie::vector<bfloat16, 16> S0; // row 0
    aie::vector<bfloat16, 16> S1; // row 1
    aie::vector<bfloat16, 16> S2; // row 2
    aie::vector<bfloat16, 16> S3; // row 3
    aie::vector<bfloat16, 16> S4; // row 4
    aie::vector<bfloat16, 16> S5; // row 5
    aie::vector<bfloat16, 16> S6; // row 6
    aie::vector<bfloat16, 16> S7; // row 7

    // taking 166 cycles = transpose + mac
    // taking 125 cycles = mac
    // q: [8,64]
    // k: [16,64]
    // q @ kt : [8,16]
    matmul_m8k64n16_vectorized_8x8x8_bf16_bf16(q,
                                               k,
                                               S0,
                                               S1,
                                               S2,
                                               S3,
                                               S4,
                                               S5,
                                               S6,
                                               S7
                                               );


    // use this: do not use y[i]
    // aie::store_v(y+0*16, S0);
    // aie::store_v(y+1*16, S1);
    // aie::store_v(y+2*16, S2);
    // aie::store_v(y+3*16, S3);
    // aie::store_v(y+4*16, S4);
    // aie::store_v(y+5*16, S5);
    // aie::store_v(y+6*16, S6);
    // aie::store_v(y+7*16, S7);
    // return;
    // above good: err = 0.003

    //event1();




    //----------------------------------------------------------------
    //event0();


    // q_scale_factor is 0: not sure what is going on: not work currently.


    // apply scale_factor
    aie::accum<accfloat, 16> SS0 = aie::mul(S0, (bfloat16)0.125f);
    S0 = SS0.template to_vector<bfloat16>();

    aie::accum<accfloat, 16> SS1 = aie::mul(S1, (bfloat16)0.125f);
    S1 = SS1.template to_vector<bfloat16>();

    aie::accum<accfloat, 16> SS2 = aie::mul(S2, (bfloat16)0.125f);
    S2 = SS2.template to_vector<bfloat16>();

    aie::accum<accfloat, 16> SS3 = aie::mul(S3, (bfloat16)0.125f);
    S3 = SS3.template to_vector<bfloat16>();


    aie::accum<accfloat, 16> SS4 = aie::mul(S4, (bfloat16)0.125f);
    S4 = SS4.template to_vector<bfloat16>();

    aie::accum<accfloat, 16> SS5 = aie::mul(S5, (bfloat16)0.125f);
    S5 = SS5.template to_vector<bfloat16>();

    aie::accum<accfloat, 16> SS6 = aie::mul(S6, (bfloat16)0.125f);
    S6 = SS6.template to_vector<bfloat16>();

    aie::accum<accfloat, 16> SS7 = aie::mul(S7, (bfloat16)0.125f);
    S7 = SS7.template to_vector<bfloat16>();


    /////////////////// on peano: also good results ///////////////

    // aie::store_v(y+0*16, S0);
    // aie::store_v(y+1*16, S1);
    // aie::store_v(y+2*16, S2);
    // aie::store_v(y+3*16, S3);
    // aie::store_v(y+4*16, S4);
    // aie::store_v(y+5*16, S5);
    // aie::store_v(y+6*16, S6);
    // aie::store_v(y+7*16, S7);
    // return;
    // above good: err = 0.003




    bfloat16 r0_max;
    bfloat16 r1_max;
    bfloat16 r2_max;
    bfloat16 r3_max;
    bfloat16 r4_max;
    bfloat16 r5_max;
    bfloat16 r6_max;
    bfloat16 r7_max;

    if(kv_padding_flag == 1) { // branch: cycle cost = 0
        r0_max = S0[0];
        r1_max = S1[0];
        r2_max = S2[0];
        r3_max = S3[0];

        r4_max = S4[0];
        r5_max = S5[0];
        r6_max = S6[0];
        r7_max = S7[0];

    } else {

        r0_max = aie::reduce_max(S0);
        r1_max = aie::reduce_max(S1);
        r2_max = aie::reduce_max(S2);
        r3_max = aie::reduce_max(S3);

        r4_max = aie::reduce_max(S4);
        r5_max = aie::reduce_max(S5);
        r6_max = aie::reduce_max(S6);
        r7_max = aie::reduce_max(S7);
    }


    // y[0] = r0_max;
    // y[1] = r1_max;
    // y[2] = r2_max;
    // y[3] = r3_max;
    // y[4] = r4_max;
    // y[5] = r5_max;
    // y[6] = r6_max;
    // y[7] = r7_max;
    // return;
    // above good



    // at least 16: 16x16 = 256bit
    // only 1st 8 elements are used
    aie::vector<bfloat16, 16> mi_new(r0_max,
                                     r1_max,
                                     r2_max,
                                     r3_max,
                                     r4_max,
                                     r5_max,
                                     r6_max,
                                     r7_max
                                     );
    mi_new = aie::max(mi, mi_new);

    // y[0] = r0_max;
    // y[1] = r1_max;
    // y[2] = r2_max;
    // y[3] = r3_max;
    // y[4] = r4_max;
    // y[5] = r5_max;
    // y[6] = r6_max;
    // y[7] = r7_max;
    // return;
    // above good



    // need broadcast first
    // aie::vector<bfloat16, 16> P0 = aie::broadcast<bfloat16, 16>((bfloat16)mi_new[0]);
    // P0 = aie::sub(S0, P0);

    // aie::vector<bfloat16, 16> P1 = aie::broadcast<bfloat16, 16>((bfloat16)mi_new[1]);
    // P1 = aie::sub(S1, P1);

    // aie::vector<bfloat16, 16> P2 = aie::broadcast<bfloat16, 16>((bfloat16)mi_new[2]);
    // P2 = aie::sub(S2, P2);

    // aie::vector<bfloat16, 16> P3 = aie::broadcast<bfloat16, 16>((bfloat16)mi_new[3]);
    // P3 = aie::sub(S3, P3);


    aie::vector<bfloat16, 16> P0 = aie::sub(S0, (bfloat16)mi_new[0]);
    aie::vector<bfloat16, 16> P1 = aie::sub(S1, (bfloat16)mi_new[1]);
    aie::vector<bfloat16, 16> P2 = aie::sub(S2, (bfloat16)mi_new[2]);
    aie::vector<bfloat16, 16> P3 = aie::sub(S3, (bfloat16)mi_new[3]);

    aie::vector<bfloat16, 16> P4 = aie::sub(S4, (bfloat16)mi_new[4]);
    aie::vector<bfloat16, 16> P5 = aie::sub(S5, (bfloat16)mi_new[5]);
    aie::vector<bfloat16, 16> P6 = aie::sub(S6, (bfloat16)mi_new[6]);
    aie::vector<bfloat16, 16> P7 = aie::sub(S7, (bfloat16)mi_new[7]);




    // aie::store_v(y+0*16, P0);
    // aie::store_v(y+1*16, P1);
    // aie::store_v(y+2*16, P2);
    // aie::store_v(y+3*16, P3);
    // aie::store_v(y+4*16, P4);
    // aie::store_v(y+5*16, P5);
    // aie::store_v(y+6*16, P6);
    // aie::store_v(y+7*16, P7);
    // return;
    // above good: err = 0.003


    // LUT-exp: have memory access issue
    // P0 = exp_bf16_func_ori(P0);
    // P1 = exp_bf16_func_ori(P1);
    // P2 = exp_bf16_func_ori(P2);
    // P3 = exp_bf16_func_ori(P3);
    // P4 = exp_bf16_func_ori(P4);
    // P5 = exp_bf16_func_ori(P5);
    // P6 = exp_bf16_func_ori(P6);
    // P7 = exp_bf16_func_ori(P7);
    //error = 1.0



    // P0 = exp_simple_bf16(P0);
    // P1 = exp_simple_bf16(P1);
    // P2 = exp_simple_bf16(P2);
    // P3 = exp_simple_bf16(P3);
    // P4 = exp_simple_bf16(P4);
    // P5 = exp_simple_bf16(P5);
    // P6 = exp_simple_bf16(P6);
    // P7 = exp_simple_bf16(P7);
    // err = 0.001  #  this will cause total error = 0.109
    /*
      total error:
      max_diff = 0.101562
      pred = 0.343750
      ref = 0.445312
      relative_err = 0.109058
     */


    P0 = exp_bf16_func(P0);
    P1 = exp_bf16_func(P1);
    P2 = exp_bf16_func(P2);
    P3 = exp_bf16_func(P3);
    P4 = exp_bf16_func(P4);
    P5 = exp_bf16_func(P5);
    P6 = exp_bf16_func(P6);
    P7 = exp_bf16_func(P7);
    // err = 0.011 #  this will cause total error = 0.11
    /*
      max_diff = 0.107422
      pred = 0.337891
      ref = 0.445312
      relative_err = 0.110883
     */

    // aie::store_v(y+0*16, P0);
    // aie::store_v(y+1*16, P1);
    // aie::store_v(y+2*16, P2);
    // aie::store_v(y+3*16, P3);
    // aie::store_v(y+4*16, P4);
    // aie::store_v(y+5*16, P5);
    // aie::store_v(y+6*16, P6);
    // aie::store_v(y+7*16, P7);
    // return;


    //event1(); // 303 cycles



    #if 1
    if(kv_padding_flag == 1) {
        {
            P0[0] = P0[0];
            P0[1] = 0;
            P0[2] = 0;
            P0[3] = 0;
            P0[4] = 0;
            P0[5] = 0;
            P0[6] = 0;
            P0[7] = 0;
            P0[8] = 0;
            P0[9] = 0;
            P0[10] = 0;
            P0[11] = 0;
            P0[12] = 0;
            P0[13] = 0;
            P0[14] = 0;
            P0[15] = 0;
        }

        {
            P1[0] = P1[0];
            P1[1] = 0;
            P1[2] = 0;
            P1[3] = 0;
            P1[4] = 0;
            P1[5] = 0;
            P1[6] = 0;
            P1[7] = 0;
            P1[8] = 0;
            P1[9] = 0;
            P1[10] = 0;
            P1[11] = 0;
            P1[12] = 0;
            P1[13] = 0;
            P1[14] = 0;
            P1[15] = 0;
        }

        {
            P2[0] = P2[0];
            P2[1] = 0;
            P2[2] = 0;
            P2[3] = 0;
            P2[4] = 0;
            P2[5] = 0;
            P2[6] = 0;
            P2[7] = 0;
            P2[8] = 0;
            P2[9] = 0;
            P2[10] = 0;
            P2[11] = 0;
            P2[12] = 0;
            P2[13] = 0;
            P2[14] = 0;
            P2[15] = 0;
        }

        {
            P3[0] = P3[0];
            P3[1] = 0;
            P3[2] = 0;
            P3[3] = 0;
            P3[4] = 0;
            P3[5] = 0;
            P3[6] = 0;
            P3[7] = 0;
            P3[8] = 0;
            P3[9] = 0;
            P3[10] = 0;
            P3[11] = 0;
            P3[12] = 0;
            P3[13] = 0;
            P3[14] = 0;
            P3[15] = 0;
        }


        {
            P4[0] = P4[0];
            P4[1] = 0;
            P4[2] = 0;
            P4[3] = 0;
            P4[4] = 0;
            P4[5] = 0;
            P4[6] = 0;
            P4[7] = 0;
            P4[8] = 0;
            P4[9] = 0;
            P4[10] = 0;
            P4[11] = 0;
            P4[12] = 0;
            P4[13] = 0;
            P4[14] = 0;
            P4[15] = 0;
        }

        {
            P5[0] = P5[0];
            P5[1] = 0;
            P5[2] = 0;
            P5[3] = 0;
            P5[4] = 0;
            P5[5] = 0;
            P5[6] = 0;
            P5[7] = 0;
            P5[8] = 0;
            P5[9] = 0;
            P5[10] = 0;
            P5[11] = 0;
            P5[12] = 0;
            P5[13] = 0;
            P5[14] = 0;
            P5[15] = 0;
        }

        {
            P6[0] = P6[0];
            P6[1] = 0;
            P6[2] = 0;
            P6[3] = 0;
            P6[4] = 0;
            P6[5] = 0;
            P6[6] = 0;
            P6[7] = 0;
            P6[8] = 0;
            P6[9] = 0;
            P6[10] = 0;
            P6[11] = 0;
            P6[12] = 0;
            P6[13] = 0;
            P6[14] = 0;
            P6[15] = 0;
        }

        {
            P7[0] = P7[0];
            P7[1] = 0;
            P7[2] = 0;
            P7[3] = 0;
            P7[4] = 0;
            P7[5] = 0;
            P7[6] = 0;
            P7[7] = 0;
            P7[8] = 0;
            P7[9] = 0;
            P7[10] = 0;
            P7[11] = 0;
            P7[12] = 0;
            P7[13] = 0;
            P7[14] = 0;
            P7[15] = 0;
        }
    }
    #endif


    // lij_hat
    bfloat16 l0_sum;
    bfloat16 l1_sum;
    bfloat16 l2_sum;
    bfloat16 l3_sum;
    bfloat16 l4_sum;
    bfloat16 l5_sum;
    bfloat16 l6_sum;
    bfloat16 l7_sum;

    if(kv_padding_flag == 1) {
        l0_sum = P0[0];
        l1_sum = P1[0];
        l2_sum = P2[0];
        l3_sum = P3[0];
        l4_sum = P4[0];
        l5_sum = P5[0];
        l6_sum = P6[0];
        l7_sum = P7[0];

    } else {

        // v16bfloat16 inp0 = P0;
        // v16bfloat16 inp1 = P1;
        // v16bfloat16 inp2 = P2;
        // v16bfloat16 inp3 = P3;


        // aie::vector<float, 16> pp0 = v16float(ups_to_v16accfloat(inp0));
        // aie::vector<float, 16> pp1 = v16float(ups_to_v16accfloat(inp1));
        // aie::vector<float, 16> pp2 = v16float(ups_to_v16accfloat(inp2));
        // aie::vector<float, 16> pp3 = v16float(ups_to_v16accfloat(inp3));

        // float k0 = 0.0f;
        // float k1 = 0.0f;
        // float k2 = 0.0f;
        // float k3 = 0.0f;

        // // for(int i=0; i<16; i++) k0 += pp0[i];
        // // for(int i=0; i<16; i++) k1 += pp1[i];
        // // for(int i=0; i<16; i++) k2 += pp2[i];
        // // for(int i=0; i<16; i++) k3 += pp3[i];

        // for(int i=0; i<16; i++) k0 += 0.1f;
        // for(int i=0; i<16; i++) k1 += 0.1f;
        // for(int i=0; i<16; i++) k2 += 0.1f;
        // for(int i=0; i<16; i++) k3 += 0.1f;

        // l0_sum = k0;
        // l1_sum = k1;
        // l2_sum = k2;
        // l3_sum = k3;


        // same as before
        //l0_sum = (bfloat16)aie::reduce_add(pp0);
        //l1_sum = (bfloat16)aie::reduce_add(pp1);
        //l2_sum = (bfloat16)aie::reduce_add(pp2);
        //l3_sum = (bfloat16)aie::reduce_add(pp3);

        l0_sum = aie::reduce_add(P0);
        l1_sum = aie::reduce_add(P1);
        l2_sum = aie::reduce_add(P2);
        l3_sum = aie::reduce_add(P3);

        l4_sum = aie::reduce_add(P4);
        l5_sum = aie::reduce_add(P5);
        l6_sum = aie::reduce_add(P6);
        l7_sum = aie::reduce_add(P7);
    }


    // need to be 256bit aligned
    // do not use y[i] = xxxx

    // aie::vector<bfloat16, 16> b0 = aie::broadcast<bfloat16, 16>(l0_sum);
    // aie::vector<bfloat16, 16> b1 = aie::broadcast<bfloat16, 16>(l1_sum);
    // aie::vector<bfloat16, 16> b2 = aie::broadcast<bfloat16, 16>(l2_sum);
    // aie::vector<bfloat16, 16> b3 = aie::broadcast<bfloat16, 16>(l3_sum);
    // aie::vector<bfloat16, 16> b4 = aie::broadcast<bfloat16, 16>(l4_sum);
    // aie::vector<bfloat16, 16> b5 = aie::broadcast<bfloat16, 16>(l5_sum);
    // aie::vector<bfloat16, 16> b6 = aie::broadcast<bfloat16, 16>(l6_sum);
    // aie::vector<bfloat16, 16> b7 = aie::broadcast<bfloat16, 16>(l7_sum);

    // aie::store_v(y + 0*16, b0);
    // aie::store_v(y + 1*16, b1);
    // aie::store_v(y + 2*16, b2);
    // aie::store_v(y + 3*16, b3);
    // aie::store_v(y + 4*16, b4);
    // aie::store_v(y + 5*16, b5);
    // aie::store_v(y + 6*16, b6);
    // aie::store_v(y + 7*16, b7);
    // return;
    // good




    // lut-exp: but error: relative_err = 0.121244
    // approx exp: error: relative_err = 0.002

    aie::vector<float, 8> lij_hat_float(l0_sum,
                                        l1_sum,
                                        l2_sum,
                                        l3_sum,
                                        l4_sum,
                                        l5_sum,
                                        l6_sum,
                                        l7_sum
                                        );

    // exp(mi - m_new)
    aie::vector<bfloat16, 16> correction; // only 1st 8 elems used
    correction = aie::sub(mi, mi_new);



    // at the very 1st compute step, y is 0, so this is not needed,
    // after 1st step, all correction is 1
    #if 0  // this may incur more cycle consumption
    bfloat16 lower_bound = -128.0f;

    if(correction[0] < lower_bound) correction[0] = -128.0f;
    if(correction[1] < lower_bound) correction[1] = -128.0f;
    if(correction[2] < lower_bound) correction[2] = -128.0f;
    if(correction[3] < lower_bound) correction[3] = -128.0f;

    if(correction[4] < lower_bound) correction[4] = -128.0f;
    if(correction[5] < lower_bound) correction[5] = -128.0f;
    if(correction[6] < lower_bound) correction[6] = -128.0f;
    if(correction[7] < lower_bound) correction[7] = -128.0f;
    #endif



    //aie::vector<bfloat16, 16> correction2 = exp_bf16_func_ori(correction);
    // LUT does not work properly now
    //correction = exp_bf16_func_ori(correction);



    correction = exp_bf16_func(correction);



    // aie::store_v(y, mi);
    // aie::store_v(y+16, mi_new);
    // aie::store_v(y+2*16, correction);

    // mi = mi_new;

    // return;
    // above good


    //----------------------------------------------------------------

    /*
     * we may need to optimize the code, because correction would be all 1 most the time.
     */
    #if 1
    aie::mask<16> eq_res_mask = aie::eq< aie::vector<bfloat16,16>, bfloat16>(correction, (bfloat16)1.0f);
    uint32_t eq_res = eq_res_mask.to_uint32() & 0x0000FFFF;

    uint8_t correction_is_one = 0;

    if(eq_res == 0x0000FFFF) {  // correction is all 1
        correction_is_one = 1;
        //event0();
    }
    #endif
    //----------------------------------------------------------------


    aie::vector<float, 8> correct_float((bfloat16)correction[0],
                                        (bfloat16)correction[1],
                                        (bfloat16)correction[2],
                                        (bfloat16)correction[3],
                                        (bfloat16)correction[4],
                                        (bfloat16)correction[5],
                                        (bfloat16)correction[6],
                                        (bfloat16)correction[7]
                                        );

    // float mul : expensive
    //aie::accum<accfloat, 8> li_new_acc = aie::mul(correct_float, li);

    aie::accum<accfloat, 8> li_new_acc(li);


    if(!correction_is_one) {
        li_new_acc = aie::mul(correct_float, li);
    }


    aie::vector<float, 8> li_new = aie::add(li_new_acc.to_vector<float>(), lij_hat_float);


    {
        // aie::vector<bfloat16, 8> kkk;

        // for(int i=0; i<8; i++) kkk[i] = (bfloat16)li_new[i];
        // aie::store_v(y, kkk);
        // return;
        //above good, but still has error
    }


    //event1(); // start at line743: 416 / 330 cycles

    //--------------------------------------------------------------

#if 1

    //event0();


    // y : [8,64], each row 64x16bit = 1024bit
    aie::vector<bfloat16, 64> y_row0 = aie::load_v<64>(y + 0*64);
    aie::vector<bfloat16, 64> y_row1 = aie::load_v<64>(y + 1*64);
    aie::vector<bfloat16, 64> y_row2 = aie::load_v<64>(y + 2*64);
    aie::vector<bfloat16, 64> y_row3 = aie::load_v<64>(y + 3*64);
    aie::vector<bfloat16, 64> y_row4 = aie::load_v<64>(y + 4*64);
    aie::vector<bfloat16, 64> y_row5 = aie::load_v<64>(y + 5*64);
    aie::vector<bfloat16, 64> y_row6 = aie::load_v<64>(y + 6*64);
    aie::vector<bfloat16, 64> y_row7 = aie::load_v<64>(y + 7*64);

    // aie::store_v(y,        y_row0);
    // aie::store_v(y + 1*64, y_row1);
    // aie::store_v(y + 2*64, y_row2);
    // aie::store_v(y + 3*64, y_row3);
    // return;

    // skip this: 990 cycles total
    // always take this: 1022 cycles total


    if(!correction_is_one) {

        aie::accum<accfloat, 64> y_row0_res = aie::mul(y_row0, correction[0]);
        aie::accum<accfloat, 64> y_row1_res = aie::mul(y_row1, correction[1]);
        aie::accum<accfloat, 64> y_row2_res = aie::mul(y_row2, correction[2]);
        aie::accum<accfloat, 64> y_row3_res = aie::mul(y_row3, correction[3]);

        aie::accum<accfloat, 64> y_row4_res = aie::mul(y_row4, correction[4]);
        aie::accum<accfloat, 64> y_row5_res = aie::mul(y_row5, correction[5]);
        aie::accum<accfloat, 64> y_row6_res = aie::mul(y_row6, correction[6]);
        aie::accum<accfloat, 64> y_row7_res = aie::mul(y_row7, correction[7]);

        // aie::accum<accfloat, 64> y_row0_res = aie::mul(y_row0, (bfloat16)1.0f);
        // aie::accum<accfloat, 64> y_row1_res = aie::mul(y_row1, (bfloat16)1.0f);
        // aie::accum<accfloat, 64> y_row2_res = aie::mul(y_row2, (bfloat16)1.0f);
        // aie::accum<accfloat, 64> y_row3_res = aie::mul(y_row3, (bfloat16)1.0f);

        y_row0 = y_row0_res.to_vector<bfloat16>(); // accum to vector
        y_row1 = y_row1_res.to_vector<bfloat16>(); // accum to vector
        y_row2 = y_row2_res.to_vector<bfloat16>(); // accum to vector
        y_row3 = y_row3_res.to_vector<bfloat16>(); // accum to vector

        y_row4 = y_row4_res.to_vector<bfloat16>(); // accum to vector
        y_row5 = y_row5_res.to_vector<bfloat16>(); // accum to vector
        y_row6 = y_row6_res.to_vector<bfloat16>(); // accum to vector
        y_row7 = y_row7_res.to_vector<bfloat16>(); // accum to vector


        //event1();
    }

    // aie::store_v(y + 0*16,         P_dot_v0);
    // aie::store_v(y + 1*16,         P_dot_v1);
    // aie::store_v(y + 2*16,         P_dot_v2);
    // aie::store_v(y + 3*16,         P_dot_v3);
    // aie::store_v(y + 4*16,         P_dot_v4);
    // aie::store_v(y + 5*16,         P_dot_v5);
    // aie::store_v(y + 6*16,         P_dot_v6);
    // aie::store_v(y + 7*16,         P_dot_v7);
    // aie::store_v(y + 8*16,         P_dot_v8);
    // aie::store_v(y + 9*16,         P_dot_v9);
    // aie::store_v(y + 10*16,         P_dot_v10);
    // aie::store_v(y + 11*16,         P_dot_v11);
    // aie::store_v(y + 12*16,         P_dot_v12);
    // aie::store_v(y + 13*16,         P_dot_v13);
    // aie::store_v(y + 14*16,         P_dot_v14);
    // aie::store_v(y + 15*16,         P_dot_v15);
    // This is correct.

    // aie::store_v(y,        pdv_row0);
    // aie::store_v(y + 1*64, pdv_row1);
    // aie::store_v(y + 2*64, pdv_row2);
    // aie::store_v(y + 3*64, pdv_row3);
    // return; // error: 0.004139 without exp func


    // aie::store_v(y+0*16, P0);
    // aie::store_v(y+1*16, P1);
    // aie::store_v(y+2*16, P2);
    // aie::store_v(y+3*16, P3);
    // aie::store_v(y+4*16, P4);
    // aie::store_v(y+5*16, P5);
    // aie::store_v(y+6*16, P6);
    // aie::store_v(y+7*16, P7);
    // return;



    // This block seems to use a lot of registers, which could cause other code
    // to have wrong results.
    /// better to put this at the end

    // because we only have 4 1024bit registers in total, and we have to
    // do data reorder in registers,
    // we cannot do rst=888 mmul, so we need to resort to rst=484

    aie::vector<bfloat16, 8*8> PB0 = aie::concat(P0.extract<8>(0), // P block 0
                                                 P1.extract<8>(0),
                                                 P2.extract<8>(0),
                                                 P3.extract<8>(0),
                                                 P4.extract<8>(0),
                                                 P5.extract<8>(0),
                                                 P6.extract<8>(0),
                                                 P7.extract<8>(0)
                                                 );

    aie::vector<bfloat16, 8*8> PB1 = aie::concat(P0.extract<8>(1), // P lock 1
                                                 P1.extract<8>(1),
                                                 P2.extract<8>(1),
                                                 P3.extract<8>(1),
                                                 P4.extract<8>(1),
                                                 P5.extract<8>(1),
                                                 P6.extract<8>(1),
                                                 P7.extract<8>(1)
                                                 );


    // aie::store_v(y+0*64, PB0);
    // aie::store_v(y+1*64, PB1);
    // return;
    // good


    //aie::vector<bfloat16, 16*64> vvv = aie::load_v<16*64>(v);
    //aie::store_v(y, vvv);
    //return;
    // good







    // P : [8,16]
    // v:  [16,64]
    // P @ V : [8,64]
    aie::vector<bfloat16, 64> PV_blk0;
    aie::vector<bfloat16, 64> PV_blk1;
    aie::vector<bfloat16, 64> PV_blk2;
    aie::vector<bfloat16, 64> PV_blk3;

    aie::vector<bfloat16, 64> PV_blk4;
    aie::vector<bfloat16, 64> PV_blk5;
    aie::vector<bfloat16, 64> PV_blk6;
    aie::vector<bfloat16, 64> PV_blk7;

    // These correct only on xchess compiler
    matmul_m8k16n64_vectorized_8x8x8_bf16_bf16(PB0, PB1, v,
                                               PV_blk0,
                                               PV_blk1,
                                               PV_blk2,
                                               PV_blk3
                                               );


    matmul_m8k16n64_vectorized_8x8x8_bf16_bf16(PB0, PB1, v + 4 * 8 * 8,
                                               PV_blk4,
                                               PV_blk5,
                                               PV_blk6,
                                               PV_blk7
                                               );

    // P : [8,16]
    // v:  [16,64]
    // P @ V : [8,64]
    aie::vector<bfloat16, 64> PV_row0;
    aie::vector<bfloat16, 64> PV_row1;
    aie::vector<bfloat16, 64> PV_row2;
    aie::vector<bfloat16, 64> PV_row3;

    aie::vector<bfloat16, 64> PV_row4;
    aie::vector<bfloat16, 64> PV_row5;
    aie::vector<bfloat16, 64> PV_row6;
    aie::vector<bfloat16, 64> PV_row7;

    // need to have distinct var names
    reshape_p_dot_v(PV_blk0, PV_blk1, PV_blk2, PV_blk3, PV_blk4, PV_blk5, PV_blk6, PV_blk7,
                    PV_row0, PV_row1, PV_row2, PV_row3, PV_row4, PV_row5, PV_row6, PV_row7);


    // aie::store_v(y+0*64, PV_row0);
    // aie::store_v(y+1*64, PV_row1);
    // aie::store_v(y+2*64, PV_row2);
    // aie::store_v(y+3*64, PV_row3);
    // aie::store_v(y+4*64, PV_row4);
    // aie::store_v(y+5*64, PV_row5);
    // aie::store_v(y+6*64, PV_row6);
    // aie::store_v(y+7*64, PV_row7);
    // return;
    // error = 0.01





    // aie::store_v(y+0*64, y_row0);
    // aie::store_v(y+1*64, y_row1);
    // aie::store_v(y+2*64, y_row2);
    // aie::store_v(y+3*64, y_row3);
    // aie::store_v(y+4*64, y_row4);
    // aie::store_v(y+5*64, y_row5);
    // aie::store_v(y+6*64, y_row6);
    // aie::store_v(y+7*64, y_row7);
    // return;


    // y = y + p @ v
    y_row0 = aie::add(y_row0, PV_row0);
    y_row1 = aie::add(y_row1, PV_row1);
    y_row2 = aie::add(y_row2, PV_row2);
    y_row3 = aie::add(y_row3, PV_row3);
    y_row4 = aie::add(y_row4, PV_row4);
    y_row5 = aie::add(y_row5, PV_row5);
    y_row6 = aie::add(y_row6, PV_row6);
    y_row7 = aie::add(y_row7, PV_row7);

    aie::store_v(y + 0*64, y_row0);
    aie::store_v(y + 1*64, y_row1);
    aie::store_v(y + 2*64, y_row2);
    aie::store_v(y + 3*64, y_row3);
    aie::store_v(y + 4*64, y_row4);
    aie::store_v(y + 5*64, y_row5);
    aie::store_v(y + 6*64, y_row6);
    aie::store_v(y + 7*64, y_row7);
    // error: 0.024126
    // good
    //---------------------------------------------


    li = li_new;
    mi = mi_new;

#endif


    //event1(); // start at line1138, 577 / 526 cycles
}



static inline void _flashv2_bfloat16(const bfloat16 *__restrict q, // [8,64]
                              const bfloat16 *__restrict k, // [16,64]
                              const bfloat16 *__restrict v, // [16,64]
                              bfloat16 *__restrict       y, // [8,64]

                              float    *__restrict l,  // [8]  : 8*32bit  = 256bit
                              bfloat16 *__restrict m,  // [16] : 16*16bit = 256bit

                              const bfloat16 q_scale_factor,

                              // single row valid: k,v padding flag
                              const int kv_padding_flag
                              )
{
    /*
     * NPU1: register operation must be aligned to 128bit
             data memory operation must be aligned to 256 bit
     */

    // 4 * 32 = 128bit, not allowed
    // load at least 8 float numbers
    // 8 * 32bit = 256bit, works on npu2

    aie::vector<float, 8>          lvec = aie::load_v<8>(l);

    // 16 * 16bit = 256bit
    aie::vector<bfloat16, 16>      mvec = aie::load_v<16>(m);



    do_kernel_flash_v2_row4_col64(q, k, v, y, lvec, mvec, q_scale_factor, kv_padding_flag);


    aie::store_v(l, lvec);
    aie::store_v(m, mvec);
}


//--------------------------------------------------------------------
// div


// 383 cycles

// 360 cycles
#if 1
static inline void do_kernel_func_bf16_div(bfloat16       *__restrict y, // [8,64]
                                    const float    *__restrict l, // [8] : 1st 8 elems used
                                    int32_t cnt)
{

    //event0();

    // at least load 256bit
    // 8 * 32bit = 256bit
    aie::vector<float, 8> lvec = aie::load_v<8>(l);

    aie::vector<float, 8> inv = aie::inv(lvec); // must be 8 elems for inv operation


    const int rows_size = 8;
    const int rows_iter = 1;


    bfloat16 *__restrict r00_ptr;
    bfloat16 *__restrict r01_ptr;


    for(int i = 0; i < rows_size; i += rows_iter)
        //chess_prepare_for_pipelining chess_loop_range(rows_size/rows_iter,)
        #ifdef OPT_PERF_ENABLED
            chess_flatten_loop
        #endif
        {

            bfloat16 scale = (bfloat16)inv[i];


            // 16bit * 64 = 1024bit, few 1024bit regs
            // 16bit * 32 = 512bit, 12 512bit regs
            // 16bit * 16 = 256 bit, 24 256bit regs

            r00_ptr = y + i * 64;
            r01_ptr = y + i * 64 + 32;
            aie::vector<bfloat16, 32> r00 = aie::load_v<32>(r00_ptr);
            aie::vector<bfloat16, 32> r01 = aie::load_v<32>(r01_ptr);

            aie::accum<accfloat, 32> res00 = aie::mul(r00, scale);
            aie::accum<accfloat, 32> res01 = aie::mul(r01, scale);

            aie::vector<bfloat16, 32> out00 = res00.to_vector<bfloat16>();
            aie::vector<bfloat16, 32> out01 = res01.to_vector<bfloat16>();

            out00.store(r00_ptr);
            out01.store(r01_ptr);
        }


    //event1();
}
#endif

//--------------------------------------------------------------------

// add inline : note for test

//--------------------------------------------------------------------



static inline void mini_block_flashv2_bfloat16(const bfloat16 *__restrict q, // [8,64]
                                        const bfloat16 *__restrict k, // [16,64]
                                        const bfloat16 *__restrict v, // [16,64]
                                        bfloat16       *__restrict y, // [8,64]

                                        // only 1st 8 elems are used
                                        float    *__restrict l, // [8]
                                        bfloat16 *__restrict m, // [16]

                                        const bfloat16 q_scale_factor,

                                        // single row valid: k,v padding flag
                                        const int kv_padding_flag
                                        )
{
    _flashv2_bfloat16(q,k,v,y, l,m, q_scale_factor, kv_padding_flag);
}


static inline void mini_block_kernel_func_div_bf16(bfloat16    *__restrict y,  // [8,64]
                                            const float *__restrict l)  // [8] : only 1st 8 elems used
{
    do_kernel_func_bf16_div(y, l, 0);
}



#endif





static inline void compute_q_normal(
                             // 64x64
                             const bfloat16 *__restrict q,
                             const bfloat16 *__restrict k,
                             bfloat16       *__restrict v,
                             bfloat16       *__restrict y,

                             // echo row: only 1st 8 elems are used
                             float    *__restrict l, // [64/8,8]
                             bfloat16 *__restrict m, // [64/8,16]

                             const bfloat16 q_scale_factor,

                             const int idx_kv,
                             const int buffer_id // 0: buffer-0, 1: buffer-1
                             )
{
    // ignore buffer-1
    if (idx_kv == 8 && buffer_id == 1) return;
    //if (idx_kv == 8) return;


    /*
      mini-block:
      q: 8x64
      k: 16x64
      v: 16x64
      o: 8x64

      one-block:
      q: 64x64
      k: 64x64
      v: 64x64
      o: 64x64
    */

    const bfloat16 *__restrict ptr_q = q;
    const bfloat16 *__restrict ptr_k = k;
    const bfloat16 *__restrict ptr_v = v;
    bfloat16 *__restrict ptr_y = y;

    float    *__restrict ptr_l = l;
    bfloat16 *__restrict ptr_m = m;


    const int rows_qy_size = 64;
    const int rows_qy_iter = 8;
    const int rows_kv_size = 64;
    const int rows_kv_iter = 16;

    const int l_iter = 8;
    const int m_iter = 16;


    if (idx_kv != 8) { // idx = 0 -> 7, buffer-0/1

        #if 1
        for(int i_q = 0; i_q < rows_qy_size;
                i_q += rows_qy_iter,
                ptr_q += (rows_qy_iter*rows_qy_size),
                ptr_y += (rows_qy_iter*rows_qy_size),
                ptr_l += l_iter,
                ptr_m += m_iter)
            chess_prepare_for_pipelining chess_loop_range(rows_qy_size/rows_qy_iter,)
            {
                ptr_k = k;
                ptr_v = v;

                for(int i_kv = 0; i_kv < rows_kv_size;
                        i_kv += rows_kv_iter,
                        ptr_k += (rows_kv_iter*rows_kv_size),
                        ptr_v += (rows_kv_iter*rows_kv_size))
                    #ifdef OPT_PERF_ENABLED
                    chess_flatten_loop
                    #endif
                        {
                            mini_block_flashv2_bfloat16(ptr_q, ptr_k, ptr_v, ptr_y,
                                                        ptr_l, ptr_m,
                                                        q_scale_factor,
                                                        0); // no kv padding
                        }
            }

        #endif


    } else { // idx = 8, buffer-0


        #if 1
        for(int i_q = 0; i_q < rows_qy_size;
                i_q += rows_qy_iter,
                ptr_q += (rows_qy_iter*rows_qy_size),
                ptr_y += (rows_qy_iter*rows_qy_size),
                ptr_l += l_iter,
                ptr_m += m_iter)
            chess_prepare_for_pipelining chess_loop_range(rows_qy_size/rows_qy_iter,)
            {
                ptr_k = k;
                ptr_v = v;

                //event0();

                for(int i_kv = 0; i_kv < rows_kv_iter; // single k,v token, only compute once
                        i_kv += rows_kv_iter,
                        ptr_k += (rows_kv_iter*rows_kv_size),
                        ptr_v += (rows_kv_iter*rows_kv_size))
                    #ifdef OPT_PERF_ENABLED
                    chess_flatten_loop
                    #endif
                        {
                            mini_block_flashv2_bfloat16(ptr_q, ptr_k, ptr_v, ptr_y,
                                                        ptr_l, ptr_m,
                                                        q_scale_factor,
                                                        1); // kv padding
                        }

                //event1();

            }

        #endif

        //event1();
    }

}




static inline void compute_q_last_token(
                                 // 64x64
                                 const bfloat16 *__restrict q,
                                 const bfloat16 *__restrict k,
                                 bfloat16       *__restrict v,
                                 bfloat16       *__restrict y,

                                 // each row: only 1st 8 elems are used
                                 float    *__restrict l, // [64/8, 8]
                                 bfloat16 *__restrict m, // [64/8, 16]

                                 const bfloat16 q_scale_factor,

                                 const int idx_kv,
                                 const int buffer_id // 0: buffer-0, 1: buffer-1
                                 )
{
    // ignore buffer-1
    if(idx_kv == 8 && buffer_id == 1) return;


    /*
      mini-block:
      q: 8x64
      k: 16x64
      v: 16x64
      o: 8x64

      one-block:
      q: 64x64
      k: 64x64
      v: 64x64
      o: 64x64
    */

    const bfloat16 *__restrict ptr_q = q;
    const bfloat16 *__restrict ptr_k = k;
    const bfloat16 *__restrict ptr_v = v;
    bfloat16 *__restrict ptr_y = y;

    float    *__restrict ptr_l = l;
    bfloat16 *__restrict ptr_m = m;



    const int rows_qy_size = 64;
    const int rows_qy_iter = 8;
    const int rows_kv_size = 64;
    const int rows_kv_iter = 16;

    const int l_iter = 8;
    const int m_iter = 16;


    if (idx_kv != 8) { // idx = 0 -> 7, buffer-0/1

        #if 1
        for(int i_q = 0; i_q < rows_qy_iter; // single q token, only compute once
                i_q += rows_qy_iter,
                ptr_q += (rows_qy_iter*rows_qy_size),
                ptr_y += (rows_qy_iter*rows_qy_size),
                ptr_l += l_iter,
                ptr_m += m_iter)
            chess_prepare_for_pipelining chess_loop_range(rows_qy_iter/rows_qy_iter,)
            {
                ptr_k = k;
                ptr_v = v;

                //event0();

                for(int i_kv = 0; i_kv < rows_kv_size;
                        i_kv += rows_kv_iter,
                        ptr_k += (rows_kv_iter*rows_kv_size),
                        ptr_v += (rows_kv_iter*rows_kv_size))
                    #ifdef OPT_PERF_ENABLED
                    chess_flatten_loop
                    #endif
                        {
                            //event0();

                            mini_block_flashv2_bfloat16(ptr_q, ptr_k, ptr_v, ptr_y,
                                                        ptr_l, ptr_m,
                                                        q_scale_factor,
                                                        0); // no kv padding

                            //event1();
                        }

                //event1();

            }

        #endif


    } else { // idx = 8, buffer-0


        //event0();

        #if 1
        for(int i_q = 0; i_q < rows_qy_iter; // single q token, only compute once
                i_q += rows_qy_iter,
                ptr_q += (rows_qy_iter*rows_qy_size),
                ptr_y += (rows_qy_iter*rows_qy_size),
                ptr_l += l_iter,
                ptr_m += m_iter)
            chess_prepare_for_pipelining chess_loop_range(rows_qy_iter/rows_qy_iter,)
            {
                ptr_k = k;
                ptr_v = v;

                //event0();

                for(int i_kv = 0; i_kv < rows_kv_iter; // single k,v token, only compute once
                        i_kv += rows_kv_iter,
                        ptr_k += (rows_kv_iter*rows_kv_size),
                        ptr_v += (rows_kv_iter*rows_kv_size))
                    #ifdef OPT_PERF_ENABLED
                    chess_flatten_loop
                    #endif
                        {
                            mini_block_flashv2_bfloat16(ptr_q, ptr_k, ptr_v, ptr_y,
                                                        ptr_l, ptr_m,
                                                        q_scale_factor,
                                                        1); // kv padding
                        }

                //event1();

            }

        #endif

        //event1();
    }

}




//--------------------------------------------------------------------
// adding inline or static: reduce latency
//--------------------------------------------------------------------



// copy: 140 cycles
template <unsigned SIZE>
static inline void do_copy_bfloat16(bfloat16           *__restrict q, // 64x64
                             const bfloat16     *__restrict v) // 64x64
{

    bfloat16 *__restrict ptr_q = q;

    bfloat16 *__restrict ptr_v = (bfloat16 *)v;

    constexpr int r = 512 / (sizeof(bfloat16) * 8);

    const bfloat16 *__restrict end = ptr_v + SIZE;


    //event0();

    for(; ptr_v < end; ptr_v += r)
      chess_prepare_for_pipelining chess_loop_range(r,) {

          aie::vector<bfloat16, r> A = aie::load_v<r>(ptr_v);

          A.store(ptr_q);
          ptr_q += r;
      }

    //event1();
}


//--------------------------------------------------------------------


// add inline : note for test

//--------------------------------------------------------------------
extern "C" {


    void copy_bfloat16_k_to_q(bfloat16       *__restrict q,
                              const bfloat16 *__restrict k,
                              const int      flag
                              )
    {
        do_copy_bfloat16<64*64>(q, k);
    }


    void compute_kernel(
                        // 64x64
                        const bfloat16 *__restrict q,
                        const bfloat16 *__restrict k,
                        bfloat16       *__restrict v,
                        bfloat16       *__restrict y,

                        // each row: only 1st 8 elems are used
                        float    *__restrict l, // [64/8, 8]
                        bfloat16 *__restrict m, // [64/8, 16]

                        // because rtp params need to be fp32 in chess compiler
                        const float q_scale_factor, // softmax scaling factor

                        const int q_last_token_flag,
                        const int idx_kv,
                        const int buffer_id // 0: buffer-0, 1: buffer-1
                        )
    {
        //if(buffer_id == 1) return;


        //do_copy_bfloat16<64*64>(y, q);
        //return;



        // if(buffer_id == 1) return;

        // event0();


        // mini_block_flashv2_bfloat16(q, k, v, y,
        //                             l, m,
        //                             q_scale_factor,
        //                             0);

        // event1();

        // // mini_block_flashv2_bfloat16(q, k + (16*64), v + (16*64), y,
        // //                             l, m,
        // //                             q_scale_factor,
        // //                             0);


        // return;




        if (q_last_token_flag == 1) {

            event0();

            compute_q_last_token(q,k,v,y, l,m, q_scale_factor, idx_kv, buffer_id);

            event1();

        } else {

            //event0();

            compute_q_normal(q,k,v,y, l,m, q_scale_factor, idx_kv, buffer_id);

            //event1();

        }
    }



    void compute_kernel_div(bfloat16       *__restrict y, // [64,64]
                            // each row: only 1st 8 elems are used
                            const float    *__restrict l, // [64/8,8]
                            const bfloat16 *__restrict m, // [64/8,16]

                            const int q_last_token_flag
                            )
    {
        bfloat16       *__restrict ptr_y = y;
        const float    *__restrict ptr_l = l;


        //return;



        // aie::vector<float, 16>     a1 = aie::load_v<16>(l);

        // //aie::vector<float, 16> a1 = aie::broadcast<float, 16>(2.0f);

        // aie::vector<bfloat16, 16>  a2;

        // for(int i=0; i<16; i++) {
        //     a2[i] = (bfloat16)a1[i];
        // }

        // a2.store(y);

        // return;



        const int rows_size = 64;
        const int rows_iter = 8;
        const int l_iter = 8;
        const int m_iter = 16;



        if(q_last_token_flag) {

            event0();

            mini_block_kernel_func_div_bf16(ptr_y, ptr_l);

            event1();

        } else {

            //event0();

            for(int i_y = 0; i_y < rows_size;
                    i_y += rows_iter,
                    ptr_y += (rows_iter*rows_size),
                    ptr_l += l_iter)
                chess_prepare_for_pipelining chess_loop_range(rows_size/rows_iter,) {


                    mini_block_kernel_func_div_bf16(ptr_y, ptr_l);

                }

            //event1();

        }

    }


}
