import torch
from torch.library import triton_op, wrap_triton
import triton
import triton.language as tl


_MAX_E4M3_VAL = torch.finfo(torch.float8_e4m3fn).max
_MAX_FP32_VAL = torch.finfo(torch.float32).max


def _get_cuda_autotune_config():
    configs = []
    for num_warps, num_stages in [
        (4, 2),
        (4, 3),
        (4, 4),
        (8, 2),
        (8, 4),
    ]:
        for BLOCK_SIZE_M in [128]:
            for BLOCK_SIZE_N in [128]:
                for BLOCK_SIZE_K in [32, 64]:
                    configs.append(
                        triton.Config(
                            {
                                "GROUP_SIZE_M" : 8,
                                "BLOCK_SIZE_M" : BLOCK_SIZE_M,
                                "BLOCK_SIZE_N" : BLOCK_SIZE_N,
                                "BLOCK_SIZE_K" : BLOCK_SIZE_K,
                            }, 
                            num_stages=num_stages, 
                            num_warps=num_warps
                        ),
                    )                        
    return configs


@triton.autotune(
    configs=_get_cuda_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def kernel_matmul_fp8_fp8(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
):
    """
    Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    accumulator_dtype = tl.float32
    
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=accumulator_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        accumulator = tl.dot(a, b, accumulator, out_dtype=accumulator_dtype)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c = accumulator#.to(tl.float16)
    # c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul_fp8_fp8(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert a.dtype == torch.float8_e4m3fn
    assert b.dtype == torch.float8_e4m3fn

    M, K = a.shape
    K, N = b.shape
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    
    # c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    kernel_matmul_fp8_fp8[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


@triton.jit
def compute_scale_from_amax(
    amax: tl. tensor,
    _MAX_E4M3_VAL: tl. constexpr,
    _MAX_FP32_VAL: tl.constexpr,
) -> tl. tensor:
    scale = tl.where(amax == 0, 1.0, _MAX_E4M3_VAL / amax)

    # 0 11111111 00000000000000000000000 = +INF в FP32
    is_inf = tl.cast(scale, tl.int32, bitcast=True) == 0x7F800000
    # берем максимально возможное FP32 число вместо +INF
    scale = tl.where(is_inf, _MAX_FP32_VAL, scale)
    
    # 1 11111111 00000000000000000000000 - используем скейлы=степени двойки
    scale_bits = tl.cast(scale, tl.uint32, bitcast=True)
    scale = tl.cast(scale_bits & 0xFF800000, tl.float32, bitcast=True)
    return scale


@triton.jit
def quant_weight_to_e4m3_kernel(
        src_ptr,
        dst_ptr,
        scale_ptr,
        M: int,
        N: int,
        BLOCK_SIZE: tl. constexpr,
        _MAX_E4M3_VAL: tl.constexpr,
        _MAX_FP32_VAL: tl. constexpr,
    ):

    # Матрица M×N, читаем блоками по BLOCK_SIZE × BLOCK_SIZE
    pid_m = tl.program_id (axis=0)
    pid_n = tl.program_id (axis=1)
    n = tl. cdiv(N, BLOCK_SIZE)

    # Какие строчки прочитать: каждая программа читаем BLOCK_SIZE подряд идущих строк
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Какие столбцы прочитать в каждой строчке
    offs_n = pid_n * BLOCK_SIZE + tl. arange(0, BLOCK_SIZE)
    #Блок BLOCK_SIZE × BLOCK_SIZE
    offs = offs_m[:, None] * N + offs_n [None, :]
    # Чтобы не выйти за границы матрицы
    mask = (offs_m[:, None] < M) & (offs_n [None, :] < N)
    src = tl.load(src_ptr + offs, mask=mask).to(tl.float32)
    
    # Считаем absmax для блока - одно число
    amax = tl.max(tl.abs(src))
    scale = compute_scale_from_amax(amax, _MAX_E4M3_VAL, _MAX_FP32_VAL)
    scale_inv = 1.0 / scale
    # Скейлим блок на scale и кастим к e4m3
    dst = (src * scale).to(dst_ptr.dtype.element_ty)
    tl. store(dst_ptr + offs, dst, mask=mask)
    tl.store(scale_ptr + pid_m * n + pid_n, scale_inv)


@triton_op("qlib::quant_weight_to_e4m3", mutates_args=("dst", "scale_dst")) 
def quant_weight_to_e4m3(
        src: torch. Tensor, 
        dst: torch.Tensor, 
        scale_dst: torch.Tensor,
        block_size: int = 128,
    ) -> None:
    assert src.is_contiguous()
    assert dst.is_contiguous ()
    assert scale_dst.is_contiguous ()

    assert len(src.shape) == 2
    M, N = src.size()
    assert src.size() == dst.size()
    assert scale_dst.size() == ((M + block_size - 1) // block_size, (N + block_size - 1) // block_size)
    
    assert dst.dtype == torch.float8_e4m3fn
    assert scale_dst.dtype == torch.float32
    
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE"]), triton.cdiv(N, meta["BLOCK_SIZE"]))

    wrap_triton(quant_weight_to_e4m3_kernel) [grid](
    src, dst, scale_dst, M=M, N=N, BLOCK_SIZE=block_size, _MAX_E4M3_VAL=_MAX_E4M3_VAL, _MAX_FP32_VAL=_MAX_FP32_VAL
)


def alloc_and_quant(x, block_size: int = 128):
    n, m = x.shape
    assert n % block_size == 0 and m % block_size == 0
    x_e4m3 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale_shape = (n // block_size, m // block_size)
    x_scale = torch.empty(*scale_shape, device=x.device, dtype=torch.float32)
    quant_weight_to_e4m3(x, x_e4m3, x_scale, block_size)
    return x_e4m3, x_scale


def _get_cuda_autotune_config_kernel_scaled_matmul_fp8_fp8():
    configs = []
    for num_warps, num_stages in [
        (4, 2),
        (4, 3),
        (4, 4),
        (8, 2),
        (8, 4),
    ]:
        for BLOCK_SIZE_K in [32, 64]:
            configs.append(
                triton.Config(
                    {
                        "GROUP_SIZE_M" : 8,
                        "BLOCK_SIZE_M" : 128,
                        "BLOCK_SIZE_N" : 128,
                        "BLOCK_SIZE_K" : 128, #BLOCK_SIZE_K,
                    }, 
                    num_stages=num_stages, 
                    num_warps=num_warps
                ),
            )                        
    return configs


@triton.autotune(
    configs=_get_cuda_autotune_config_kernel_scaled_matmul_fp8_fp8(),
    key=['M', 'N', 'K'],
)
@triton.jit
def kernel_scaled_matmul_fp8_fp8(
        a_ptr, b_ptr, c_ptr,
        a_scale_ptr, b_scale_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
):
    """
    Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    accumulator_dtype = tl.float32
    
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    a_s_base = a_scale_ptr + pid_m * num_pid_k
    b_s_base = b_scale_ptr + pid_n


    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=accumulator_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        a_s = tl.load(a_s_base + k)
        b_s = tl.load(b_s_base + k * num_pid_n)
        s = a_s * b_s

        accumulator += tl.dot(a, b, out_dtype=accumulator_dtype) * s
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c = accumulator#.to(tl.float16)
    # c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def scaled_matmul_fp8_fp8(a, b, a_scale, b_scale):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert a.dtype == torch.float8_e4m3fn
    assert b.dtype == torch.float8_e4m3fn

    M, K = a.shape
    K, N = b.shape
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    
    # c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    kernel_scaled_matmul_fp8_fp8[grid](
        a, b, c,
        a_scale, b_scale,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c