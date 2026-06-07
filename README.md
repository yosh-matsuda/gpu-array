# gpu-array: Make GPU programming more modern C++ friendly

gpu-array is a header-only C++20 library that brings RAII and Range-based abstractions to GPU memory management and data layouts, enabling code safety and performance optimizations with zero overhead. By abstracting away raw pointers and memory layouts, gpu-array allows developers to focus on algorithm logic rather than resource bookkeeping.

Maximum GPU performance with Modern C++ syntax.

[![Tests](https://github.com/yosh-matsuda/gpu-array/actions/workflows/tests.yml/badge.svg)](https://github.com/yosh-matsuda/gpu-array/actions/workflows/tests.yml)

## Features

*   Smart pointer-like wrappers:
    *   Full RAII (Resource Acquisition Is Initialization) support for GPU memory management, ensuring automatic cleanup.
*   Performance-Oriented Memory Layouts:
    *   AoS to SoA Conversion: Converting Array-of-Structures (AoS) to Structure-of-Arrays (SoA) to ensure coalesced memory access for maximum GPU throughput. AoS stores data as contiguous structures, while SoA separates each field into its own array for better memory access patterns.
    *   Jagged Array Wrappers: Manage multi-dimensional data with varying row lengths using a single, efficient 1-D memory allocation and optimized for cache locality and performance.
*   C++20 Integration:
    *   Compatible with modern standards, including ranges and iterator concepts even for GPU kernel code.
    *   Range adapters for grid-stride access patterns (e.g., block-thread, grid-thread, grid-block, etc.).
    *   GPU-ready range views for indexing (`views::enumerate`) and lock-step traversal (`views::zip`).
*   Dual backend:
    *   Support for NVIDIA CUDA and AMD HIP.
*   Header-only library and no external dependencies.

### Requirements

gpu-array requires a C++20 compiler and either the CUDA or HIP development toolkit.

<details>
<summary>Supported and tested toolkit/compiler combinations</summary>

The following toolkit/compiler combinations are supported and tested:

| Backend | Toolkit | Tested Compiler |
| --- | --- | --- |
| CUDA | 12.6.3 | GCC 13, Clang 16-18 |
| CUDA | 12.8.1 | GCC 13-14, Clang 16-19 |
| CUDA | 12.9.1 | **Not Supported** |
| CUDA | 13.0.2 | GCC 13-15, Clang 16-20 |
| CUDA | 13.1.1 | GCC 13-15, Clang 16-21 |
| CUDA | 13.2.0 | GCC 13-15, Clang 16-21 |
| ROCm/HIP NVIDIA | 6.2.4 + CUDA 12.8.1 | Clang 18 |
| ROCm/HIP NVIDIA | 6.4.4 + CUDA 12.8.1 | Clang 18 |
| ROCm/HIP NVIDIA | 7.0.3 + CUDA 12.8.1 | Clang 18 |
| ROCm/HIP NVIDIA | 7.1.1 + CUDA 12.8.1 | Clang 18 |
| ROCm/HIP NVIDIA | 7.2.4 + CUDA 13.2.0 | Clang 18 |
| ROCm/HIP AMD | 6.2.4 | AMD Clang 18 |
| ROCm/HIP AMD | 6.4.4 | AMD Clang 19 |
| ROCm/HIP AMD | 7.0.3 | AMD Clang 20 |
| ROCm/HIP AMD | 7.1.1 | AMD Clang 21 |
| ROCm/HIP AMD | 7.2.4 | AMD Clang 22 |

</details>

CUDA 12.9.1 is not supported because `nvcc` 12.9 is known to segfault while compiling gpu-array tests.

With ROCm/HIP, do not put `std::ranges` concept constraints directly on `__global__`
function templates. Current ROCm compilers can reject otherwise valid constrained
kernel launches during overload resolution. Check such constraints in an ordinary
host wrapper and launch an unconstrained `__global__` kernel from that wrapper.

The practical C++ compiler floor is GCC 13 or Clang 16. CUDA's official host
compiler tables allow older compiler majors, but gpu-array relies on C++20
ranges and related library support, so older compilers are outside the supported
range.

## Quick Start

### Installation

As a header-only library, you can simply copy the `include` directory to your project. If you are using CMake, you can add the following lines to your `CMakeLists.txt`:

```cmake
add_subdirectory(path/to/gpu-array)
target_link_libraries(your_target PRIVATE gpu_array::gpu_array)
```

Alternatively, you can use CMake's `FetchContent` module instead of manually downloading the library:

```cmake
include(FetchContent)
FetchContent_Declare(
    gpu_array
    GIT_REPOSITORY https://github.com/yosh-matsuda/gpu-array.git
    GIT_TAG v0.3.0
)
FetchContent_MakeAvailable(gpu_array)
target_link_libraries(your_target PRIVATE gpu_array::gpu_array)
```

### Example: Device memory management with smart pointers

gpu-array provides several smart pointer-like classes to manage GPU memory, including `array` and `managed_array` for arrays with range concepts, and `value` and `managed_value` for single value pointers.  
These classes automatically handle memory allocation and deallocation on the GPU. The `managed_` variants use unified memory, allowing seamless access from both host and device.

```cpp
#include <cooperative_groups.h>
#include <gpu_array.hpp>
#include <iostream>

using namespace gpu_array;

// Example kernel: initialize all elements
template <std::ranges::input_range T>
__global__ void kernel(T array)
{
    for (auto& v : array | views::grid_thread_stride)
        v += 1;
}

void example()
{
    // Allocate managed (or unmanaged) memory for 1024 integers
    auto array = managed_array<int>(1024);

    // Launch kernel to set values
    kernel<<<1, 128>>>(array);

    // Wrapper for cudaDeviceSynchronize/hipDeviceSynchronize
    api::gpuDeviceSynchronize();

    // Print results
    for (const auto& v: array) std::cout << v << " ";
}
```

gpu-array also provides safe initialization for memory allocation. For memory accessible only from the GPU (`array` and `value`), it checks at compile time that the type is safe for `memcpy` and initializes the allocated memory using the specified method. For Unified memory accessible from both the GPU and CPU (`managed_array` and `managed_value`), it constructs each element by calling its constructor.

### Example: Conversion from host to device memory and vice versa

Arrays and values classes can be easily converted from and to C++ containers (e.g., `std::vector`, `std::array`). The data is copied from host to device during construction.

```cpp
#include <gpu_array.hpp>
#include <vector>

using namespace gpu_array;

void example()
{
    // Create vector on host
    auto vec = std::vector<int>(100);
    for (auto i = 0; auto& v: vec) v = i++;

    // Convert from host vector to device array
    auto array = managed_array(vec);

    // Call kernel to perform operations on GPU
    // ...

    // Convert from device array to host vector
    vec = array.to<std::vector>();
}
```

### Example: Grid-stride range adapters

The kernel code can utilize C++20 range views for grid-stride access patterns (so-called grid-stride loop). gpu-array provides several [Range Adapter Closure Object](https://en.cppreference.com/w/cpp/named_req/RangeAdaptorClosureObject.html) such as `views::block_thread_stride`, `views::grid_thread_stride`, and `views::grid_block_stride` to facilitate this without any overhead. The following example demonstrates how to achieve memory coalescing when initializing nested arrays using grid-stride access.

```cpp
#include <gpu_array.hpp>
#include <iostream>
#include <vector>

using namespace gpu_array;

// Example kernel: initialize nested array
template <std::ranges::input_range T>
requires std::ranges::input_range<std::ranges::range_value_t<T>>
__global__ void kernel_example(T array)
{
    for (auto& a : array | views::grid_block_stride)
    {
        for (auto& v : a | views::block_thread_stride)
        {
            v *= 2;
        }
    }

    /* The above code is equivalent to the following:
    using namespace cooperative_groups;
    const auto grid = this_grid();
    const auto block = this_thread_block();
    for (auto i = grid.block_rank(); i < array.size(); i += grid.num_blocks())
    {
        for (auto j = block.thread_rank(); j < array[i].size(); j += block.size())
        {
            array[i][j] *= 2;
        }
    }
    */
}

void example()
{
    // Create nested vector on host
    auto vec_vec = std::vector(256, std::vector<int>(1024, 1));

    // Convert from nested host vector to nested device array
    auto nested_array = managed_array(vec_vec);

    // Launch kernel to initialize nested array
    kernel_example<<<32, 256>>>(nested_array);
    api::gpuDeviceSynchronize();

    // Print results
    for (const auto& inner_array : nested_array)
    {
        for (const auto& v : inner_array) std::cout << v << " ";
        std::cout << std::endl;
    }
}
```

The view adapters can also be composed with `views::enumerate` and `views::zip`. Use `views::enumerate` when the original index is part of the computation, and `views::zip` when multiple ranges should be traversed in lock step. Apply stride adapters after these sized views to distribute the resulting elements across GPU threads.

```cpp
// Use indices inside a grid-stride loop
__global__ void kernel_with_index(managed_array<int> array)
{
    for (auto&& [i, v] : array | views::enumerate | views::grid_thread_stride)
    {
        v += static_cast<int>(i);
    }
}

// Traverse two arrays together; iteration stops at the shorter size
__global__ void add_kernel(managed_array<int> lhs, managed_array<int> rhs)
{
    for (auto&& [x, y] : views::zip(lhs, rhs) | views::grid_thread_stride)
    {
        x += y;
    }
}
```

### Example: AoS and SoA

gpu-array supports both Array of structures (AoS) and Structure of arrays (SoA) for memory layout optimization via `array` and `structure_of_arrays` classes, respectively. The memory layout comparison between `array` (AoS) and `structure_of_arrays` (SoA) is as follows:

![array of structure vs. structure_of_arrays](https://github.com/user-attachments/assets/219085eb-80c7-44e5-9e3b-6607bd8174bf)

 In either case, gpu-array provides a structure retrieval interface via array indices. Thus, `structure_of_arrays<tuple-derived>` can be used as a drop-in replacement for `array<tuple-derived>` with optimizing memory layout without altering your algorithm's implementation.

```cpp
#include <gpu_array.hpp>
#include <vector>

using namespace gpu_array;

// gpu_array::tuple is a lightweight std::tuple-like type for GPU device code.
// gpu_array::tuple (or std::tuple) or its derived struct can be used as structure type
// The below example shows a tuple-derived struct with three members and their accessors
template <typename... Ts>
requires (sizeof...(Ts) == 3)
struct CustomTuple : public tuple<Ts...>
{
    using tuple<Ts...>::tuple;
    using tuple<Ts...>::operator=;
    __host__ __device__ auto& get_a() { return get<0>(*this); }
    __host__ __device__ auto& get_b() { return get<1>(*this); }
    __host__ __device__ auto& get_c() { return get<2>(*this); }
};
using Struct = CustomTuple<int, float, double>;

// Example kernel: process both AoS and SoA
template <typename T>
__global__ void kernel_example(T array)
{
    for (auto&& v : array | views::grid_thread_stride)
    {
        // Access structure members for both AoS and SoA
        v.get_a() *= 2;
        v.get_b() *= 2.0f;
        v.get_c() *= 2.0;
    }
}

void example()
{
    // Create vector of structures
    auto vec = std::vector<Struct>(100, {1, 2.0f, 3.0});

    // Array of structures (AoS): single array for entire structure
    auto aos = managed_array<Struct>(vec);
    kernel_example<<<1, 32>>>(aos);
    api::gpuDeviceSynchronize();

    // Structure of arrays (SoA): multiple arrays for each member internally
    auto soa = managed_structure_of_arrays<Struct>(vec);
    kernel_example<<<1, 32>>>(soa);
    api::gpuDeviceSynchronize();
}
```

> [!TIP]
> Which is better, AoS or SoA? It depends on the access pattern to the structure members within a warp, the size of the structure, and the number of available registers. If all threads in a warp access the same member of the structure, SoA is generally better for maximizing coalesced memory access. However, if each thread accesses entire structures or different members, AoS may be more efficient due to better cache locality. Benchmarking both layouts with representative workloads is recommended to determine the optimal choice for your specific use case. Whichever, **gpu-array makes it easy to switch between AoS and SoA without changing the access interface**.

### Example: Jagged array

gpu-array provides `jagged_array` class to manage multi-dimensional array with varying row lengths, using a **single memory allocation to maximize coalescing access**.
This behaves like a wrapper for `managed_array` or `managed_structure_of_arrays` with multi-dimensional indexing. The `jagged_array` is constructed from a 1-D array with sizes or multi-dimensional container (e.g., `std::vector<std::vector<T>>`).

The logical and physical data layout of `jagged_array` is as follows:

![data layout of jagged array](https://github.com/user-attachments/assets/7773537d-7259-4d2c-a695-8572906a6057)

```cpp
#include <gpu_array.hpp>
#include <iostream>
#include <vector>

using namespace gpu_array;

// Example kernel: modify all elements
template <std::ranges::input_range T>
__global__ void kernel(T array)
{
    for (auto& v : array | views::grid_thread_stride)
        v *= 2;
}

auto vec = std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
auto vec_vec = std::vector<std::vector<int>>{{0}, {1, 2}, {3, 4, 5}, {6, 7, 8, 9}};

void example()
{
    // Create jagged array from nested std::vector
    auto jarray = jagged_array<managed_array<int>>(vec_vec);
    // Equivalent to the above:
    // auto jarray = jagged_array<managed_array<int>>({1, 2, 3, 4}, vec);

    // Launch kernel to re-set values
    kernel<<<1, 32>>>(jarray);
    api::gpuDeviceSynchronize();

    // Access each row and each element
    for (std::size_t i = 0; i < jarray.num_rows(); ++i)
    {
        for (std::size_t j = 0; j < jarray.size(i); ++j)
        {
            std::cout << jarray[{i, j}] << " ";
        }
        std::cout << std::endl;
    }
}
```

### Zero-overhead

gpu-array is designed to have zero-overhead compared to traditional raw pointer usage. The following equivalent kernels are covered by a CUDA-only, release-like PTX/ptxas regression test. For these representative patterns, the normalized PTX instructions and ptxas resource usage match the raw-pointer baseline, so the tested abstractions add no extra loop-body instructions, register pressure, stack usage, or spills.

```cpp
// Traditional raw pointer kernel
__global__ void func0(int* data, std::uint32_t size)
{
    const auto block = cooperative_groups::this_thread_block();
    for (std::uint32_t i = block.thread_rank(); i < size; i += block.size())
    {
        data[i] = 1;
    }
}

// Using managed_array
__global__ void func1(managed_array<int, std::uint32_t> arr)
{
    const auto block = cooperative_groups::this_thread_block();
    for (std::uint32_t i = block.thread_rank(); i < arr.size(); i += block.size())
    {
        arr[i] = 1;
    }
}

// Using managed_array with range adapters
__global__ void func3(managed_array<int, std::uint32_t> arr)
{
    for (auto& v : arr | views::block_thread_stride)
    {
        v = 1;
    }
}
```

<details>

<summary>A representative PTX assembly body generated for the above kernels is as follows:</summary>

```text
// Function Definition: _Z5func3... (Mangled C++ name for a template function)
// It accepts a 'managed_array' struct by value, which is 24 bytes in size.
.visible .entry _Z5func3IN7gpu_array13managed_arrayIijEEEvT_(
    .param .align 8 .b8 _Z5func3IN7gpu_array13managed_arrayIijEEEvT__param_0[24]
)
{
    // Register Declarations
    .reg .pred  %p<3>;   // Predicate registers for logic
    .reg .b32  %r<17>;  // 32-bit integer registers
    .reg .b64  %rd<6>;  // 64-bit registers for pointers/addresses

    // --- Extracting values from the Struct Parameter ---
    // The struct is laid out in memory:
    // Offset 0: Array Size (32-bit)
    // Offset 8: Base Pointer (64-bit)
    ld.param.u64  %rd1, [_Z5func3IN7gpu_array13managed_arrayIijEEEvT__param_0+8]; // Load data pointer
    ld.param.u32  %r5, [_Z5func3IN7gpu_array13managed_arrayIijEEEvT__param_0];    // Load array size N

    // --- Calculate 1D Thread Index (%r16) ---
    // Formula: idx = (tid.z * ntid.y + tid.y) * ntid.x + tid.x
    mov.u32  %r2, %ntid.y;              // %r2 = blockDim.y
    mov.u32  %r9, %tid.z;               // %r9 = threadIdx.z
    mov.u32  %r10, %tid.y;              // %r10 = threadIdx.y
    mad.lo.s32  %r11, %r2, %r9, %r10;   // %r11 = (blockDim.y * threadIdx.z) + threadIdx.y

    mov.u32  %r3, %ntid.x;              // %r3 = blockDim.x
    mov.u32  %r12, %tid.x;              // %r12 = threadIdx.x
    mad.lo.s32  %r16, %r11, %r3, %r12;  // %r16 = (%r11 * blockDim.x) + threadIdx.x

    // --- Initial Boundary Check ---
    setp.ge.u32  %p1, %r16, %r5;        // if (idx >= N)
    @%p1 bra  $L__BB2_3;                // Exit if initial index is out of bounds

    // --- Calculate Stride (Total Threads in Block) ---
    // stride = blockDim.x * blockDim.y * blockDim.z
    mul.lo.s32  %r13, %r3, %r2;         // %r13 = blockDim.x * blockDim.y
    mov.u32  %r14, %ntid.z;             // %r14 = blockDim.z
    mul.lo.s32  %r6, %r13, %r14;        // %r6 = total threads (stride)

    // Convert generic pointer to global memory pointer
    cvta.to.global.u64  %rd3, %rd1;     // %rd3 = global(data)

// --- Main Loop: Grid-Stride Writing ---
$L__BB2_2:
    // Memory address calculation: addr = base + (idx * 4)
    mul.wide.u32  %rd4, %r16, 4;        // Calculate 64-bit byte offset
    add.s64  %rd5, %rd3, %rd4;          // Add offset to base pointer

    // Store integer 1 into memory
    mov.u32  %r15, 1;                   // Value = 1
    st.global.u32  [%rd5], %r15;        // data[idx] = 1;

    // Increment index by stride
    add.s32  %r16, %r16, %r6;           // idx += stride;

    // Loop Condition
    setp.lt.u32  %p2, %r16, %r5;        // if (idx < N)
    @%p2 bra  $L__BB2_2;                // Repeat if within bounds

$L__BB2_3:
    ret;                                // Kernel end
}
```

</details>

The automated `zero_overhead_ptx` CTest also checks `views::grid_thread_stride`, `views::enumerate`, `views::zip`, `views::zip | views::enumerate`, and `value` / `managed_value` dereference. It compares strict normalized opcode sequences where possible, and uses non-prologue opcode profiles when raw pointers and wrapper objects have different parameter-unpacking prologues.

### Tips

To reduce the number of registers used by the kernel, consider setting `size_type` to `std::uint32_t` instead of `default_size_type (= std::size_t)` when declaring GPU pointer types. For example, use `managed_array<T, std::uint32_t>` when the number of elements is less than 2<sup>32</sup>. To change `default_size_type` to `std::uint32_t`, define the `GPU_USE_32BIT_SIZE_TYPE_DEFAULT` macro before including `gpu_array.hpp`.

### Backends selection

Define `ENABLE_HIP` macro to use HIP backend. By default, CUDA backend is used. You can define this in your CMakeLists.txt or compiler flags.

## Reference

Full API reference is available in [docs/reference.md](docs/reference.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
