import torch
# os.environ['TRITON_INTERPRET'] = '1'
import triton
import triton.language as tl


@triton.jit
def to_int4(v):
    sign = (v & 0x8) != 0
    mag  = (v & 0x7).to(tl.int8)
    return tl.where(sign, -mag, mag)


def get_cuda_autotune_config():
    return [
        # triton.Config({"GROUP_SIZE_M" : 8}, num_stages=1, num_warps=1),
        # triton.Config({"GROUP_SIZE_M" : 8}, num_stages=1, num_warps=2),
        # triton.Config({"GROUP_SIZE_M" : 8}, num_stages=1, num_warps=4),
        # triton.Config({"GROUP_SIZE_M" : 8}, num_stages=1, num_warps=8),

        # triton.Config({"GROUP_SIZE_M" : 8}, num_stages=2, num_warps=1),
        # triton.Config({"GROUP_SIZE_M" : 8}, num_stages=2, num_warps=2),
        # triton.Config({"GROUP_SIZE_M" : 8}, num_stages=2, num_warps=4),
        # triton.Config({"GROUP_SIZE_M" : 8}, num_stages=2, num_warps=8),

        # triton.Config({"GROUP_SIZE_M" : 8}, num_stages=3, num_warps=1),
        # triton.Config({"GROUP_SIZE_M" : 8}, num_stages=3, num_warps=2),
        # triton.Config({"GROUP_SIZE_M" : 8}, num_stages=3, num_warps=4),
        # triton.Config({"GROUP_SIZE_M" : 8}, num_stages=3, num_warps=8),

        # triton.Config({"GROUP_SIZE_M" : 8}, num_stages=4, num_warps=1),
        # triton.Config({"GROUP_SIZE_M" : 8}, num_stages=4, num_warps=2),
        # triton.Config({"GROUP_SIZE_M" : 8}, num_stages=4, num_warps=4),
        triton.Config({"GROUP_SIZE_M" : 1}, num_stages=4, num_warps=8),
        ]

@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=[],
)
@triton.jit
def matmul_trellis_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,  #
        # Matrix dimensions
        B, IN, OUT,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # pid = tl.program_id(axis=0)
    # grid_n = tl.cdiv(OUT, 16)
    # pid_m = pid // grid_n
    # pid_n = pid % grid_n


    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(B, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(OUT, 16)
    
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # -----------------------------------------------------------
    # Add some integer bound assumptions.
    # This helps to guide integer analysis in the backend to optimize
    # load/store offset address calculation
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % B
    offs_ak = tl.arange(0, 16)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak)
    
    offs_low = (pid_n * 64 + tl.arange(0, 64)) % (OUT * 4)
    offs_high = (pid_n * 64 + (tl.arange(1, 65) % 64)) % (OUT * 4)  

    # offs_high = 1 + tl.arange(0, 64)
    # offs_high = (pid_n * 64 + tl.where(offs_high==64, 0, offs_high)) % (OUT * 4)

    b_ptrs_low = b_ptr + offs_low
    b_ptrs_high = b_ptr + offs_high
        
    accumulator_dtype = tl.float32 #tl.float16 #tl.float32
    accumulator = tl.zeros((BLOCK_SIZE_M, 16), dtype=accumulator_dtype)
    for k in range(0, tl.cdiv(IN, 16)):
        a = tl.load(a_ptrs, mask=offs_ak[None, :] < IN - k * 16, other=0.0)
        bits_low = tl.load(b_ptrs_low)#, mask=offs_k[:, None] < IN - k * 16, other=0.0)
        bits_high = tl.load(b_ptrs_high)
        codes = bits_low.to(tl.uint16) | (bits_high.to(tl.uint16) << 8)
        codes = ((codes.to(tl.uint32) * 34038481) >> 9).to(tl.uint16)

        val0 = (codes >> 12) & 0xF
        val1 = (codes >> 8) & 0xF
        val2 = (codes >> 4) & 0xF
        val3 = codes & 0xF

        w0 = to_int4(val0)
        w1 = to_int4(val1)
        w2 = to_int4(val2)
        w3 = to_int4(val3)

        # print("w0", w0)
        # print("w1", w1)
        # print("w2", w2)
        # print("w3", w3)

        w01 = tl.join(w0, w2)
        w23 = tl.join(w1, w3)
        w = tl.join(w01, w23)

        #print(w.shape)

        # w01 = tl.cat(w0, w1, can_reorder=True)
        # w23 = tl.cat(w2, w3, can_reorder=True)
        # w = tl.cat(w01, w23, can_reorder=True)
        
        #w = tl.reshape(w, 4, 64)
        #w = tl.trans(w, 1, 0)
        #print()
        w = tl.reshape(w, 16, 16)
        #print(w)
        #raise
        # We accumulate along the K dimension.
        # print(w)
        accumulator = tl.dot(a, w.to(tl.float16), accumulator, out_dtype=accumulator_dtype)
        # Advance the ptrs to the next K block.
        a_ptrs += 16 * stride_ak
        b_ptrs_high += 1 * stride_bk
        b_ptrs_low += 1 * stride_bk


    c = accumulator.to(tl.float16)
    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * 16 + tl.arange(0, 16)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < B) & (offs_cn[None, :] < OUT)
    tl.store(c_ptrs, c, mask=c_mask)


def trellis_matmul_triton(a, b_compressed):
    b_compressed = b_compressed.reshape(b_compressed.shape[0], -1)
    
    assert a.shape[1] == b_compressed.shape[0] * 16, "Incompatible dimensions"
    assert b_compressed.is_contiguous(), "Matrix B_compressed must be contiguous"

    B, IN = a.shape
    OUT = b_compressed.shape[-1] // 4

    # Init out ptr
    c = torch.empty((B, OUT), device=b_compressed.device, dtype=torch.float16)

    # 1D launch kernel where each block gets its own program.
    BLOCK_SIZE_M = 256
    BLOCK_SIZE_K = 16
    BLOCK_SIZE_N = 16
    
    grid = lambda META: (triton.cdiv(B, BLOCK_SIZE_M) * triton.cdiv(OUT, BLOCK_SIZE_N), )
    matmul_trellis_kernel[grid](
        a, b_compressed, c,  #
        B, IN, OUT,  #
        a.stride(0), 
        a.stride(1),  #
        b_compressed.stride(0), 
        b_compressed.stride(1),  #
        c.stride(0), 
        c.stride(1),  #
        BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N
    )
    return c