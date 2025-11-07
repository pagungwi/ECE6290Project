//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <bits/stdc++.h>
#include <boost/program_options.hpp>
#include <cstdint>
#include <fstream>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdfloat>
#include <vector>


#include <algorithm>
#include <bits/stdc++.h>
#include <boost/program_options.hpp>
#include <cmath>
#include <fstream>
#include <optional>
#include <ostream>
#include <stdfloat>



#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "test_utils.h"

#include "xrt_test_wrapper.h"


#include "common.h"




//------------------------------------
#include <algorithm>
#include <limits>


#include <xtensor/containers/xarray.hpp>
#include <xtensor/io/xnpy.hpp>

#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/generators/xrandom.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/views/xdynamic_view.hpp>
#include <xtensor/views/xview.hpp>


//------------------------------------




using bfloat16_t = std::bfloat16_t;

typedef bfloat16_t bf16;



namespace po = boost::program_options;


using namespace std;





//-------------------------------------
// each tile in kernel computation

int d_head = 64;

int q_dim0 = 64;
int q_dim1 = d_head;

int v_dim0 = 64;
int v_dim1 = d_head;

int k_dim0 = 64;
int k_dim1 = d_head;

int o_dim0 = 64;
int o_dim1 = d_head;




int B = 1;
int H = 1;
int L = 1025;

int out_check_size = B * H * 1025 * d_head;



int L_PROCESS = 1024;

int q_global_dim0 = B * H * L;
int k_global_dim0 = B * H * L;
int v_global_dim0 = B * H * L;

//int o_global_dim0 = B * H * (L_PROCESS + 4 * o_dim0);

// this can be used for trace, need to have more empty space 
int o_global_dim0 = B * H * (L_PROCESS + 4 * o_dim0 * o_dim1);

int o_global_dim1 = d_head;

int q_global_rem_dim0 = B * H * (L - L_PROCESS);
int k_global_rem_dim0 = B * H * (L - L_PROCESS);
int v_global_rem_dim0 = B * H * (L - L_PROCESS);


int q_block_dim0  = d_head;
int q_block_dim1  = d_head;

int k_block_dim0  = d_head;
int k_block_dim1  = d_head;

int v_block_dim0  = d_head;
int v_block_dim1  = d_head;

int o_block_dim0  = d_head;
int o_block_dim1  = d_head;



//-----------------------------------------------------------





//-----------------------------------------------------------

double compute_relative_error(const std::vector<bf16>& y_true,
                              const std::vector<bf16>& y_pred)
{
    double r_err = 0.0;

    double ref_all = 0.0;

    size_t n = y_pred.size();


    double max_diff = -100000.0f;
    double pred, ref;


    for (size_t i = 0; i < n; ++i) {

        double diff = (double)y_pred[i] - (double)y_true[i];

        r_err += pow(diff, 2);

        ref_all += pow((double)y_true[i], 2);

        double abs_diff = fabs(diff);

        if(abs_diff > max_diff) {
            max_diff = abs_diff;
            pred = y_pred[i];
            ref = y_true[i];
        }

    }

    r_err = sqrt(r_err);

    ref_all = sqrt(ref_all);


    printf("max_diff = %f \n", max_diff);
    printf("pred = %f \n", pred);
    printf("ref = %f \n", ref);


    return (r_err / ref_all);
}



//-----------------------------------------------------------




void print_bfloat16_hex(uint16_t bf16_value) {
    std::cout << "bfloat16 hex = 0x"
              << std::hex << std::setw(4) << std::setfill('0')
              << bf16_value << std::endl;
}


void print_matrix1(bfloat16_t* a, int m, int k)
{
    printf("\n");
    for(int i=0; i<m*k; i++) {
        printf("%f ", ((float)a[i]));
    }
    printf("\n");
}

void print_matrix2(bfloat16_t* a, int m, int k)
{
    printf("\n");
    for(int i=0; i<m; i++) {

        //printf("line-%d \n", i);

        for(int j=0; j<k; j++) {
            //printf("%f ", ((float)a[i*k + j]));

            std::cout<<a[i*k + j]<<" ";

            //print_bfloat16_hex(a[i*k + j]);


            #if 0
            if( ((float)a[i*k + j]) != 2.0f) {
                printf("output error ! \n");
                exit(0);
            }
            #endif

        }
        printf("\n");
    }
    printf("\n");
    printf("----------------------\n");

}

void print_matrix3(std::vector<bfloat16_t> a, int m, int k)
{
    printf("\n");
    for(int i=0; i<m; i++) {
        //printf("line-%d \n", i);
        for(int j=0; j<k; j++) {
            printf("%f ", ((float)a[i*k + j]));
        }
        printf("\n\n");
    }
    printf("\n");
    printf("----------------------\n");
}
//-----------------------------------------------------------

void print_exp_test(int n)
{
    n = 16 * 16;

    float p = 0.125;
    float s = -0.0125;

    for(int i=0; i<n; i++) {

        std::bfloat16_t b = (bf16)s;

        s += p;

        std::bfloat16_t ref = exp(b);

        //std::cout<< b << " ";
        std::cout<< ref << " ";
    }
    printf("\n");
    printf("----------------------\n");
}


void print_bf16_vec(std::vector<bf16> v)
{
    for(int i=0; i<v.size(); i++) {
        print_bfloat16_hex(v[i]);
        std::cout<< (float)v[i] << std::endl;
    }
}


void print_bf16(bf16* v, int k)
{
    for(int i=0; i<k; i++) {
        print_bfloat16_hex(v[i]);
        std::cout<< (float)v[i] << std::endl;
    }
}


std::vector<bf16> load_npy_tensor(string filename)
{
    std::vector<bf16> v;

    auto arr = xt::load_npy<uint16_t>(filename);


    int total_size = arr.size();

    int sel_size = B * H * L * d_head;


    if(sel_size > total_size) {
        printf("total size: %d \n", total_size);
        printf("select size: %d \n", sel_size);

        printf("numpy file size: error \n");
        exit(0);
    }


    for (int i=0; i < sel_size; i++) {

        bf16 res;

        uint16_t tmp = arr[i + 12 * 3 * 1025 * 64];

        std::memcpy(&res, &tmp, sizeof(uint16_t));

        v.push_back(res);
    }

    return v;
}




std::vector<bf16> load_npy2_tensor(string filename)
{
    std::vector<bf16> v;

    auto arr = xt::load_npy<uint32_t>(filename);

    //printf("arr.size() = %d \n", arr.size());

    int total_size = arr.size();

    int sel_size = B * H * L * d_head;


    if(sel_size > total_size) {
        printf("numpy file size: error \n");
        exit(0);
    }


    for (int i=0; i<sel_size; i++) {

        uint32_t tmp = arr[i] << 16;

        float f;

        std::memcpy(&f, &tmp, sizeof(float));

        bf16 res = (bf16)f;

        v.push_back(res);
    }

    return v;
}


std::vector<bf16> load_npy_f32_tensor(string filename)
{
    std::vector<bf16> v;

    auto arr = xt::load_npy<float>(filename);

    for (auto k : arr) {

        bf16 res = (bf16)k;

        v.push_back(res);
    }

    return v;
}


//-----------------------------------------------------------






int main(int argc, const char *argv[])
{
  args myargs = parse_args(argc, argv);



  int verbosity = myargs.verbosity;
  int do_verify = myargs.do_verify;

  int n_iterations = myargs.n_iterations;
  int n_warmup_iterations = myargs.n_warmup_iterations;


  int trace_bytes = myargs.trace_size;

  //printf("tace bytes: %d \n", trace_bytes);





  //--------------------------------------------------
  std::vector<bf16> vec_q;
  std::vector<bf16> vec_qk; // with q at 1st
  std::vector<bf16> vec_v;
  std::vector<bf16> vec_o;

  #if 0
  std::vector<bf16> q_tensor = load_npy2_tensor("./py/npy_save_13b_16h/13b_16h_q.npy");
  std::vector<bf16> k_tensor = load_npy2_tensor("./py/npy_save_13b_16h/13b_16h_k.npy");
  std::vector<bf16> v_tensor = load_npy2_tensor("./py/npy_save_13b_16h/13b_16h_v.npy");
  std::vector<bf16> o_tensor = load_npy2_tensor("./py/npy_save_13b_16h/13b_16h_o.npy");
  #endif


  //std::vector<bf16> q_1b1h_tensor = load_npy_tensor("./py/npy_save_single_head/q_uint16.npy");
  //std::vector<bf16> o_1b1h_tensor = load_npy_tensor("./py/npy_save_single_head/o_uint16.npy");


  #if 0
  std::vector<bf16> q_tensor = load_npy_tensor("./py/npy_save_single_head/q_uint16.npy");
  std::vector<bf16> k_tensor = load_npy_tensor("./py/npy_save_single_head/k_uint16.npy");
  std::vector<bf16> v_tensor = load_npy_tensor("./py/npy_save_single_head/v_uint16.npy");
  std::vector<bf16> o_tensor = load_npy_tensor("./py/npy_save_single_head/o_uint16.npy");
  #endif


  #if 1
  std::vector<bf16> q_tensor = load_npy_tensor("./py/save/q_all_uint16.npy");
  std::vector<bf16> k_tensor = load_npy_tensor("./py/save/k_all_uint16.npy");
  std::vector<bf16> v_tensor = load_npy_tensor("./py/save/v_all_uint16.npy");
  std::vector<bf16> o_tensor = load_npy_tensor("./py/save/o_all_uint16.npy");
  #endif


  // std::cout<<q_tensor.size()<<std::endl;
  // matmul_common::print_matrix<bf16>(q_tensor, 64);
  // return 0;




  //double err =  compute_relative_error(q_tensor, q_1b1h_tensor);

  //printf("err = %f \n", err);
  //return 0;






  for(auto a : q_tensor) {
      vec_qk.push_back(a);
  }

  // std::cout<<q_tensor.size()<<std::endl;
  // matmul_common::print_matrix<bf16>(q_tensor, 64);

  // return 0;



  for(auto a : k_tensor) {
      vec_qk.push_back(a);
  }

  for(auto a : v_tensor) {
      vec_v.push_back(a);
  }

  for(auto a : o_tensor) {
      vec_o.push_back(a);
  }


  std::vector<bf16> oi_tensor = load_npy_f32_tensor("py/flash_v2/oi_float.npy");
  // for(auto a : oi_tensor) {
  //     std::cout<<a<<" ";
  // }
  // return 0;







  // ------------------------------------------------------

  int IN_SIZE_QK  = vec_qk.size();
  int IN_SIZE_V   = vec_v.size();

  // must append: (4-1) * o_dim0 * o_dim1
  int OUT_SIZE_O  = o_global_dim0 * o_global_dim1;

  int qk_bytes = IN_SIZE_QK * sizeof(bf16);
  int v_bytes  = IN_SIZE_V  * sizeof(bf16);
  int o_bytes  = OUT_SIZE_O * sizeof(bf16);


  // add trace data space
  int out_bytes = o_bytes + trace_bytes;

  // ------------------------------------------------------


  // Load instruction sequence
  std::vector<uint32_t> instr_v = test_utils::load_instr_binary(myargs.instr);

  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // ------------------------------------------------------
  // Get device, load the xclbin & kernel and register them
  // ------------------------------------------------------
  xrt::device device;
  xrt::kernel kernel;



  auto start0 = std::chrono::high_resolution_clock::now();


  test_utils::init_xrt_load_kernel(device, kernel, verbosity,
                                   myargs.xclbin, myargs.kernel);

  auto stop0 = std::chrono::high_resolution_clock::now();
  float npu_time0 =
      std::chrono::duration_cast<std::chrono::microseconds>(stop0 - start0).count();
  std::cout<<"################ load xclbin time: "<<npu_time0<<" us"<<std::endl;





  // ------------------------------------------------------
  // Initialize input/ output buffer sizes and sync them
  // ------------------------------------------------------
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));

  auto bo_in_QK = xrt::bo(device, qk_bytes, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));

  auto bo_in_V  = xrt::bo(device, v_bytes, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  // Assumes trace will only be added to bo_out
  auto bo_out   = xrt::bo(device, out_bytes, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));


  //--------------------------------------------------

  // QK
  bfloat16_t *bufInQK = bo_in_QK.map<bfloat16_t *>();
  memcpy(bufInQK, vec_qk.data(), qk_bytes);


  // V
  bfloat16_t *bufInV = bo_in_V.map<bfloat16_t *>();
  memcpy(bufInV, vec_v.data(), v_bytes);

  //--------------------------------------------------



  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";




  auto start1 = std::chrono::high_resolution_clock::now();


  // Initialize instruction buffer
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));


  // Sync buffers to update input buffer values
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);



  auto stop1 = std::chrono::high_resolution_clock::now();
  float npu_time1 =
      std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1).count();
  std::cout<<"################ load instr time: "<<npu_time1<<" us"<<std::endl;



  bo_in_QK.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in_V.sync(XCL_BO_SYNC_BO_TO_DEVICE);






  // ------------------------------------------------------
  // Initialize run configs
  // ------------------------------------------------------
  unsigned num_iter = n_iterations + n_warmup_iterations;
  float npu_time_total = 0;
  float npu_time_min = 9999999;
  float npu_time_max = 0;

  int errors = 0;

  // ------------------------------------------------------
  // Main run loop
  // ------------------------------------------------------
  for (unsigned iter = 0; iter < num_iter; iter++) {
  //for (unsigned iter = 0; iter < 1; iter++) {

    // Run kernel
    if (verbosity >= 1)
        std::cout << "Running Kernel.\n";

    auto start = std::chrono::high_resolution_clock::now();
    unsigned int opcode = 3;

    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_in_QK, bo_in_V, bo_out);

    run.wait();
    auto stop = std::chrono::high_resolution_clock::now();
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    if (iter < n_warmup_iterations) {
        continue;
    }


    bf16 *bufOut = bo_out.map<bf16 *>();

    if (trace_bytes > 0) {
        test_utils::write_out_trace(((char *)bufOut) + o_bytes, trace_bytes,
                                    myargs.trace_file);
    }

    float npu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

    npu_time_total += npu_time;
    npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
    npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;
  }



  std::cout<<"==================="<<std::endl;



  bf16 *bufOut = bo_out.map<bf16 *>();


  std::vector<bf16> vec_out;
  // // //for(int i=0; i < (1 * 64 * 64); i++) vec_out.push_back(bufOut[i + 0*64*64]);
  // // //for(int i=0; i < (1 * 64 * 64); i++) vec_out.push_back(k_tensor[i + 0*64*64]);

  //for(int i=0; i < (B * H * L * d_head); i++) vec_out.push_back(bufOut[i]);
  for(int i=0; i < (out_check_size); i++) vec_out.push_back(bufOut[i]);

  // print_matrix3(oi_tensor, 9, 64);
  // printf("------------------------------ \n");
  //print_matrix3(vec_out, 1025, 64);

  //return 0;

  //print_matrix2(bufOut + (13 * 4 * 4), 4, 4);


  // float abc = 0;
  // for(int i=0; i<16; i++) {
  //     abc += oi_tensor[i + 1*16];
  // }

  // printf("abc = %f \n", abc);
  // return 0;


  printf("PC refer: \n");
  //matmul_common::print_matrix<bf16>(oi_tensor, 64);

  matmul_common::print_matrix<bf16>(vec_o, 64);
  //matmul_common::print_matrix<bf16>(vec_q, 64);

  std::cout<<"==================="<<std::endl;

  printf("NPU: \n");
  matmul_common::print_matrix<bf16>(vec_out, 64);

  std::cout<<"==================="<<std::endl;


  double relative_err =  compute_relative_error(vec_o, vec_out);
  //double relative_err =  compute_relative_error(q_tensor, vec_out);

  //double relative_err =  compute_relative_error(oi_tensor, vec_out);
  printf("relative_err = %f \n", relative_err);

  std::cout<<"==================="<<std::endl;




  //print_lut();
  // //exit(0);


  // for(int i=0; i<512; i++) {
  //     // if(exp_ilut_ab_bf16[i] != bufOut[i]) {
  //     //     std::cout<<"error: "<<exp_ilut_ab_bf16[i]<<" - "<<bufOut[i]<<std::endl;
  //     // }

  //     if(exp_flut_ab_bf16[i] != bufOut[i]) {
  //         std::cout<<"error: "<<exp_flut_ab_bf16[i]<<" - "<<bufOut[i]<<std::endl;
  //     }
  // }
  // printf("\n");



  // ------------------------------------------------------
  // Print verification and timing results
  // ------------------------------------------------------

  // TODO - Mac count to guide gflops
  float macs = 0;

  std::cout << std::endl
            << "Avg NPU time: " << npu_time_total / n_iterations << "us."
            << std::endl;
  if (macs > 0)
    std::cout << "Avg NPU gflops: "
              << macs / (1000 * npu_time_total / n_iterations) << std::endl;

  std::cout << std::endl
            << "Min NPU time: " << npu_time_min << "us." << std::endl;
  if (macs > 0)
    std::cout << "Max NPU gflops: " << macs / (1000 * npu_time_min)
              << std::endl;

  std::cout << std::endl
            << "Max NPU time: " << npu_time_max << "us." << std::endl;
  if (macs > 0)
    std::cout << "Min NPU gflops: " << macs / (1000 * npu_time_max)
              << std::endl;



}
