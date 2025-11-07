module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    %shim_noc_tile_4_0 = aie.tile(4, 0)
    %shim_noc_tile_5_0 = aie.tile(5, 0)
    %shim_noc_tile_6_0 = aie.tile(6, 0)
    %shim_noc_tile_7_0 = aie.tile(7, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %mem_tile_2_1 = aie.tile(2, 1)
    %mem_tile_3_1 = aie.tile(3, 1)
    %mem_tile_4_1 = aie.tile(4, 1)
    %mem_tile_5_1 = aie.tile(5, 1)
    %mem_tile_6_1 = aie.tile(6, 1)
    %mem_tile_7_1 = aie.tile(7, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_2 = aie.tile(3, 2)
    %tile_4_2 = aie.tile(4, 2)
    %tile_5_2 = aie.tile(5, 2)
    %tile_6_2 = aie.tile(6, 2)
    %tile_7_2 = aie.tile(7, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_1_3 = aie.tile(1, 3)
    %tile_2_3 = aie.tile(2, 3)
    %tile_3_3 = aie.tile(3, 3)
    %tile_4_3 = aie.tile(4, 3)
    %tile_5_3 = aie.tile(5, 3)
    %tile_6_3 = aie.tile(6, 3)
    %tile_7_3 = aie.tile(7, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_1_4 = aie.tile(1, 4)
    %tile_2_4 = aie.tile(2, 4)
    %tile_3_4 = aie.tile(3, 4)
    %tile_4_4 = aie.tile(4, 4)
    %tile_5_4 = aie.tile(5, 4)
    %tile_6_4 = aie.tile(6, 4)
    %tile_7_4 = aie.tile(7, 4)
    %tile_0_5 = aie.tile(0, 5)
    %tile_1_5 = aie.tile(1, 5)
    %tile_2_5 = aie.tile(2, 5)
    %tile_3_5 = aie.tile(3, 5)
    %tile_4_5 = aie.tile(4, 5)
    %tile_5_5 = aie.tile(5, 5)
    %tile_6_5 = aie.tile(6, 5)
    %tile_7_5 = aie.tile(7, 5)
    %buffer_0_1 = aie.buffer(%mem_tile_0_1) : memref<64x64xbf16> 
    %lock_0_1 = aie.lock(%mem_tile_0_1, 0) {init = 1 : i32}
    %lock_0_1_0 = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32}
    %buffer_0_1_1 = aie.buffer(%mem_tile_0_1) : memref<64x64xbf16> 
    %lock_0_1_2 = aie.lock(%mem_tile_0_1, 2) {init = 1 : i32}
    %lock_0_1_3 = aie.lock(%mem_tile_0_1, 3) {init = 0 : i32}
    %buffer_0_1_4 = aie.buffer(%mem_tile_0_1) : memref<64x64xbf16> 
    %lock_0_1_5 = aie.lock(%mem_tile_0_1, 4) {init = 1 : i32}
    %lock_0_1_6 = aie.lock(%mem_tile_0_1, 5) {init = 0 : i32}
    %buffer_0_1_7 = aie.buffer(%mem_tile_0_1) : memref<64x64xbf16> 
    %lock_0_1_8 = aie.lock(%mem_tile_0_1, 6) {init = 1 : i32}
    %lock_0_1_9 = aie.lock(%mem_tile_0_1, 7) {init = 0 : i32}
    %buffer_0_1_10 = aie.buffer(%mem_tile_0_1) : memref<64x64xbf16> 
    %lock_0_1_11 = aie.lock(%mem_tile_0_1, 8) {init = 1 : i32}
    %lock_0_1_12 = aie.lock(%mem_tile_0_1, 9) {init = 0 : i32}
    %buffer_0_1_13 = aie.buffer(%mem_tile_0_1) : memref<64x64xbf16> 
    %lock_0_1_14 = aie.lock(%mem_tile_0_1, 10) {init = 1 : i32}
    %lock_0_1_15 = aie.lock(%mem_tile_0_1, 11) {init = 0 : i32}
    %buffer_0_1_16 = aie.buffer(%mem_tile_0_1) : memref<64x64xbf16> 
    %lock_0_1_17 = aie.lock(%mem_tile_0_1, 12) {init = 1 : i32}
    %lock_0_1_18 = aie.lock(%mem_tile_0_1, 13) {init = 0 : i32}
    %buffer_0_1_19 = aie.buffer(%mem_tile_0_1) : memref<64x64xbf16> 
    %lock_0_1_20 = aie.lock(%mem_tile_0_1, 14) {init = 1 : i32}
    %lock_0_1_21 = aie.lock(%mem_tile_0_1, 15) {init = 0 : i32}
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      aie.use_lock(%lock_0_1_11, AcquireGreaterEqual)
      aie.dma_bd(%buffer_0_1_10 : memref<64x64xbf16>)
      aie.use_lock(%lock_0_1_12, Release)
      aie.next_bd ^bb3
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb5)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%lock_0_1_14, AcquireGreaterEqual)
      aie.dma_bd(%buffer_0_1_13 : memref<64x64xbf16>)
      aie.use_lock(%lock_0_1_15, Release)
      aie.next_bd ^bb1
    ^bb4:  // 2 preds: ^bb2, ^bb6
      aie.use_lock(%lock_0_1_12, AcquireGreaterEqual)
      aie.dma_bd(%buffer_0_1_10 : memref<64x64xbf16>, 0, 4096, [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>])
      aie.use_lock(%lock_0_1_11, Release)
      aie.next_bd ^bb6
    ^bb5:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb8)
    ^bb6:  // pred: ^bb4
      aie.use_lock(%lock_0_1_15, AcquireGreaterEqual)
      aie.dma_bd(%buffer_0_1_13 : memref<64x64xbf16>, 0, 4096, [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>])
      aie.use_lock(%lock_0_1_14, Release)
      aie.next_bd ^bb4
    ^bb7:  // 2 preds: ^bb5, ^bb9
      aie.use_lock(%lock_0_1_17, AcquireGreaterEqual)
      aie.dma_bd(%buffer_0_1_16 : memref<64x64xbf16>)
      aie.use_lock(%lock_0_1_18, Release)
      aie.next_bd ^bb9
    ^bb8:  // pred: ^bb5
      %3 = aie.dma_start(MM2S, 1, ^bb10, ^bb11)
    ^bb9:  // pred: ^bb7
      aie.use_lock(%lock_0_1_20, AcquireGreaterEqual)
      aie.dma_bd(%buffer_0_1_19 : memref<64x64xbf16>)
      aie.use_lock(%lock_0_1_21, Release)
      aie.next_bd ^bb7
    ^bb10:  // 2 preds: ^bb8, ^bb12
      aie.use_lock(%lock_0_1_18, AcquireGreaterEqual)
      aie.dma_bd(%buffer_0_1_16 : memref<64x64xbf16>, 0, 4096, [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>])
      aie.use_lock(%lock_0_1_17, Release)
      aie.next_bd ^bb12
    ^bb11:  // pred: ^bb8
      %4 = aie.dma_start(S2MM, 2, ^bb13, ^bb14)
    ^bb12:  // pred: ^bb10
      aie.use_lock(%lock_0_1_21, AcquireGreaterEqual)
      aie.dma_bd(%buffer_0_1_19 : memref<64x64xbf16>, 0, 4096, [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>])
      aie.use_lock(%lock_0_1_20, Release)
      aie.next_bd ^bb10
    ^bb13:  // 2 preds: ^bb11, ^bb17
      aie.use_lock(%lock_0_1, AcquireGreaterEqual)
      aie.dma_bd(%buffer_0_1 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_0_1_0, Release)
      aie.next_bd ^bb15
    ^bb14:  // pred: ^bb11
      %5 = aie.dma_start(MM2S, 2, ^bb18, ^bb19)
    ^bb15:  // pred: ^bb13
      aie.use_lock(%lock_0_1_2, AcquireGreaterEqual)
      aie.dma_bd(%buffer_0_1_1 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_0_1_3, Release)
      aie.next_bd ^bb16
    ^bb16:  // pred: ^bb15
      aie.use_lock(%lock_0_1_5, AcquireGreaterEqual)
      aie.dma_bd(%buffer_0_1_4 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_0_1_6, Release)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%lock_0_1_8, AcquireGreaterEqual)
      aie.dma_bd(%buffer_0_1_7 : memref<64x64xbf16>, 0, 4096)
      aie.use_lock(%lock_0_1_9, Release)
      aie.next_bd ^bb13
    ^bb18:  // 2 preds: ^bb14, ^bb22
      aie.use_lock(%lock_0_1_0, AcquireGreaterEqual)
      aie.dma_bd(%buffer_0_1 : memref<64x64xbf16>)
      aie.use_lock(%lock_0_1, Release)
      aie.next_bd ^bb20
    ^bb19:  // pred: ^bb14
      aie.end
    ^bb20:  // pred: ^bb18
      aie.use_lock(%lock_0_1_3, AcquireGreaterEqual)
      aie.dma_bd(%buffer_0_1_1 : memref<64x64xbf16>)
      aie.use_lock(%lock_0_1_2, Release)
      aie.next_bd ^bb21
    ^bb21:  // pred: ^bb20
      aie.use_lock(%lock_0_1_6, AcquireGreaterEqual)
      aie.dma_bd(%buffer_0_1_4 : memref<64x64xbf16>)
      aie.use_lock(%lock_0_1_5, Release)
      aie.next_bd ^bb22
    ^bb22:  // pred: ^bb21
      aie.use_lock(%lock_0_1_9, AcquireGreaterEqual)
      aie.dma_bd(%buffer_0_1_7 : memref<64x64xbf16>)
      aie.use_lock(%lock_0_1_8, Release)
      aie.next_bd ^bb18
    }
    %buffer_0_2 = aie.buffer(%tile_0_2) : memref<64xf32> 
    %buffer_0_2_22 = aie.buffer(%tile_0_2) : memref<64xbf16> 
    %buffer_0_2_23 = aie.buffer(%tile_0_2) : memref<64x64xbf16> 
    %buffer_0_2_24 = aie.buffer(%tile_0_2) : memref<16x64xbf16> 
    %rtp_config_buffer_col0_ct0 = aie.buffer(%tile_0_2) {sym_name = "rtp_config_buffer_col0_ct0"} : memref<16xi32> 
    %rtp_params_buffer_col0_ct0 = aie.buffer(%tile_0_2) {sym_name = "rtp_params_buffer_col0_ct0"} : memref<16xf32> 
    %buffer_0_2_25 = aie.buffer(%tile_0_2) : memref<64x64xbf16> 
    %lock_0_2 = aie.lock(%tile_0_2, 0) {init = 1 : i32}
    %lock_0_2_26 = aie.lock(%tile_0_2, 1) {init = 0 : i32}
    %buffer_0_2_27 = aie.buffer(%tile_0_2) : memref<64x64xbf16> 
    %lock_0_2_28 = aie.lock(%tile_0_2, 2) {init = 1 : i32}
    %lock_0_2_29 = aie.lock(%tile_0_2, 3) {init = 0 : i32}
    %buffer_0_2_30 = aie.buffer(%tile_0_2) : memref<64x64xbf16> 
    %lock_0_2_31 = aie.lock(%tile_0_2, 4) {init = 1 : i32}
    %lock_0_2_32 = aie.lock(%tile_0_2, 5) {init = 0 : i32}
    %buffer_0_2_33 = aie.buffer(%tile_0_2) : memref<64x64xbf16> 
    %lock_0_2_34 = aie.lock(%tile_0_2, 6) {init = 1 : i32}
    %lock_0_2_35 = aie.lock(%tile_0_2, 7) {init = 0 : i32}
    %buffer_0_2_36 = aie.buffer(%tile_0_2) : memref<64x64xbf16> 
    %lock_0_2_37 = aie.lock(%tile_0_2, 8) {init = 1 : i32}
    %lock_0_2_38 = aie.lock(%tile_0_2, 9) {init = 0 : i32}
    %buffer_0_2_39 = aie.buffer(%tile_0_2) : memref<64x64xbf16> 
    %lock_0_2_40 = aie.lock(%tile_0_2, 10) {init = 1 : i32}
    %lock_0_2_41 = aie.lock(%tile_0_2, 11) {init = 0 : i32}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      aie.use_lock(%lock_0_2_31, AcquireGreaterEqual)
      aie.dma_bd(%buffer_0_2_30 : memref<64x64xbf16>)
      aie.use_lock(%lock_0_2_32, Release)
      aie.next_bd ^bb3
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb5)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%lock_0_2_34, AcquireGreaterEqual)
      aie.dma_bd(%buffer_0_2_33 : memref<64x64xbf16>)
      aie.use_lock(%lock_0_2_35, Release)
      aie.next_bd ^bb1
    ^bb4:  // 2 preds: ^bb2, ^bb4
      aie.use_lock(%lock_0_2_29, AcquireGreaterEqual)
      aie.dma_bd(%buffer_0_2_27 : memref<64x64xbf16>, 0, 4096) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 7>}
      aie.use_lock(%lock_0_2_28, Release)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb8
      aie.use_lock(%lock_0_2_37, AcquireGreaterEqual)
      aie.dma_bd(%buffer_0_2_36 : memref<64x64xbf16>)
      aie.use_lock(%lock_0_2_38, Release)
      aie.next_bd ^bb8
    ^bb7:  // pred: ^bb5
      aie.end
    ^bb8:  // pred: ^bb6
      aie.use_lock(%lock_0_2_40, AcquireGreaterEqual)
      aie.dma_bd(%buffer_0_2_39 : memref<64x64xbf16>)
      aie.use_lock(%lock_0_2_41, Release)
      aie.next_bd ^bb6
    }
    func.func private @kernel_func_zero_bf16_o(memref<64x64xbf16>)
    func.func private @kernel_func_zero_float_l(memref<64xf32>)
    func.func private @kernel_func_neg_inf_bf16_m(memref<64xbf16>)
    func.func private @copy_bfloat16_k_to_q(memref<64x64xbf16>, memref<64x64xbf16>)
    func.func private @compute_kernel(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>, memref<64xf32>, memref<64xbf16>, memref<64x64xbf16>, memref<16x64xbf16>, f32, i32, i32, i32)
    func.func private @compute_kernel_div(memref<64x64xbf16>, memref<64xf32>, memref<64xbf16>, i32)
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        aie.use_lock(%lock_0_2_32, AcquireGreaterEqual)
        func.call @copy_bfloat16_k_to_q(%buffer_0_2_25, %buffer_0_2_30) : (memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.use_lock(%lock_0_2_31, Release)
        aie.use_lock(%lock_0_2_35, AcquireGreaterEqual)
        aie.use_lock(%lock_0_2_34, Release)
        aie.use_lock(%lock_0_2_32, AcquireGreaterEqual)
        aie.use_lock(%lock_0_2_31, Release)
        aie.use_lock(%lock_0_2_35, AcquireGreaterEqual)
        aie.use_lock(%lock_0_2_34, Release)
        func.call @kernel_func_zero_bf16_o(%buffer_0_2_27) : (memref<64x64xbf16>) -> ()
        func.call @kernel_func_zero_float_l(%buffer_0_2) : (memref<64xf32>) -> ()
        func.call @kernel_func_neg_inf_bf16_m(%buffer_0_2_22) : (memref<64xbf16>) -> ()
        aie.use_lock(%lock_0_2_28, AcquireGreaterEqual)
        %c0_42 = arith.constant 0 : index
        %0 = memref.load %rtp_config_buffer_col0_ct0[%c0_42] : memref<16xi32>
        %c0_43 = arith.constant 0 : index
        %1 = memref.load %rtp_params_buffer_col0_ct0[%c0_43] : memref<16xf32>
        %c0_44 = arith.constant 0 : index
        %c9 = arith.constant 9 : index
        %c1_45 = arith.constant 1 : index
        scf.for %arg1 = %c0_44 to %c9 step %c1_45 {
          aie.use_lock(%lock_0_2_32, AcquireGreaterEqual)
          aie.use_lock(%lock_0_2_38, AcquireGreaterEqual)
          %2 = arith.index_cast %arg1 : index to i32
          %c0_i32 = arith.constant 0 : i32
          func.call @compute_kernel(%buffer_0_2_25, %buffer_0_2_30, %buffer_0_2_36, %buffer_0_2_27, %buffer_0_2, %buffer_0_2_22, %buffer_0_2_23, %buffer_0_2_24, %1, %0, %2, %c0_i32) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>, memref<64xf32>, memref<64xbf16>, memref<64x64xbf16>, memref<16x64xbf16>, f32, i32, i32, i32) -> ()
          aie.use_lock(%lock_0_2_31, Release)
          aie.use_lock(%lock_0_2_37, Release)
          aie.use_lock(%lock_0_2_35, AcquireGreaterEqual)
          aie.use_lock(%lock_0_2_41, AcquireGreaterEqual)
          %3 = arith.index_cast %arg1 : index to i32
          %c1_i32 = arith.constant 1 : i32
          func.call @compute_kernel(%buffer_0_2_25, %buffer_0_2_33, %buffer_0_2_39, %buffer_0_2_27, %buffer_0_2, %buffer_0_2_22, %buffer_0_2_23, %buffer_0_2_24, %1, %0, %3, %c1_i32) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>, memref<64xf32>, memref<64xbf16>, memref<64x64xbf16>, memref<16x64xbf16>, f32, i32, i32, i32) -> ()
          aie.use_lock(%lock_0_2_34, Release)
          aie.use_lock(%lock_0_2_40, Release)
        }
        func.call @compute_kernel_div(%buffer_0_2_27, %buffer_0_2, %buffer_0_2_22, %0) : (memref<64x64xbf16>, memref<64xf32>, memref<64xbf16>, i32) -> ()
        aie.use_lock(%lock_0_2_29, Release)
      }
      aie.end
    } {link_with = "kernels.a", stack_size = 4096 : i32}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %mem_tile_0_1, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 2, %shim_noc_tile_0_0, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_0_2, DMA : 1)
    aie.packet_flow(7) {
      aie.packet_source<%tile_0_2, DMA : 0>
      aie.packet_dest<%mem_tile_0_1, DMA : 2>
    }
    memref.global "public" @qk_in : memref<131200xbf16>
    memref.global "public" @v_in : memref<4096xbf16>
    memref.global "public" @o_out : memref<65600xbf16>
    aie.shim_dma_allocation @qk_in(MM2S, 0, 0)
    aie.shim_dma_allocation @v_in(MM2S, 1, 0)
    aie.shim_dma_allocation @o_out(S2MM, 0, 0)
    aiex.runtime_sequence @sequence(%arg0: memref<2050x64xbf16>, %arg1: memref<1025x64xbf16>, %arg2: memref<1025x64xbf16>) {
      aiex.npu.rtp_write(@rtp_config_buffer_col0_ct0, 0, 0)
      aiex.npu.rtp_write(@rtp_params_buffer_col0_ct0, 0, 0)
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 1, 1, 73728][0, 0, 0, 1]) {id = 0 : i64, metadata = @v_in} : memref<1025x64xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 16384][0, 0, 0, 1]) {id = 1 : i64, metadata = @qk_in} : memref<2050x64xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 65600][1, 1, 1, 73728][0, 0, 0, 1]) {id = 2 : i64, metadata = @qk_in} : memref<2050x64xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][8, 8, 8, 8][512, 8, 64, 1]) {id = 3 : i64, issue_token = true, metadata = @o_out} : memref<1025x64xbf16>
      aiex.npu.dma_wait {symbol = @o_out}
    }
  }
}

