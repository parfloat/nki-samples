"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

NKI-based implementation for matrix multiplication tutorial.
Converted from PyTorch version to pure NKI/numpy.

"""

import os
import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

from matrix_multiplication_nki_kernels import (
    nki_matmul_basic_,
    nki_matmul_tiled_,
    nki_matmul_hoist_load_,
    nki_matmul_block_free_dimension_,
    nki_matmul_fully_optimized_
)

WORKING_DIRECTORY = "/home/ubuntu/matmul/"


def numpy_matmul(lhs, rhs):
    """NumPy reference implementation for matrix multiplication"""
    return np.matmul(lhs, rhs)


if __name__ == "__main__":

    # Test the small workload with basic kernel
    # Shape: lhs_small [64, 128], rhs_small [128, 512]
    lhs_small = (np.random.random_sample([64, 128]) - 0.5) * 2
    rhs_small = (np.random.random_sample([128, 512]) - 0.5) * 2

    lhs_small = nl.static_cast(lhs_small, nl.bfloat16)
    rhs_small = nl.static_cast(rhs_small, nl.bfloat16)

    # Run NKI kernel (expects lhs transposed)
    output_small = nki_matmul_basic_(lhs_small.T, rhs_small)

    # Run numpy reference
    output_small_np = numpy_matmul(lhs_small, rhs_small)

    # Compare results
    print("Checking correctness of nki_matmul_basic")
    if np.allclose(output_small_np, output_small, atol=1e-2, rtol=1e-2):
        print("NKI and NumPy match")
    else:
        print("NKI and NumPy differ")

    # Test the large workload with tiled kernels
    # Shape: lhs [4096, 1024], rhs [1024, 2048]
    lhs = (np.random.random_sample([4096, 1024]) - 0.5) * 2
    rhs = (np.random.random_sample([1024, 2048]) - 0.5) * 2

    lhs = nl.static_cast(lhs, nl.bfloat16)
    rhs = nl.static_cast(rhs, nl.bfloat16)

    # Run numpy reference
    output_np = numpy_matmul(lhs, rhs)

    def check_match(nki_func, mode="profile"):
        """Test and optionally profile an NKI matmul kernel"""
        if mode == "profile":
            print(f"Profiling {nki_func.__name__}")
            profile_func = nki.profile(
                working_directory=os.path.join(WORKING_DIRECTORY, f"{nki_func.__name__}-profiles"),
                save_neff_name='file.neff',
                save_trace_name='profile.ntff',
                profile_nth=2)(nki_func)
            output = profile_func(lhs.T, rhs)
        else:
            output = nki_func(lhs.T, rhs)

        if np.allclose(output_np, output, atol=1e-2, rtol=1e-2):
            print("NKI and NumPy match")
        else:
            print("NKI and NumPy differ")

    print("Checking correctness of nki_matmul_tiled")
    check_match(nki_matmul_tiled_)

    print("Checking correctness of nki_matmul_hoist_load")
    check_match(nki_matmul_hoist_load_)

    print("Checking correctness of nki_matmul_block_free_dimension")
    check_match(nki_matmul_block_free_dimension_)

    print("Checking correctness of nki_matmul_fully_optimized")
    check_match(nki_matmul_fully_optimized_)
