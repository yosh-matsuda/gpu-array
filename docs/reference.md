# Reference

Detailed API reference for gpu-array.

## `gpu_array::tuple`

`gpu_array::tuple` is a lightweight tuple implementation intended for GPU device code. It provides a `std::tuple`-like interface, including `gpu_array::get`, `gpu_array::apply`, and `std::tuple_size` / `std::tuple_element` support, and is designed to be used as a device-friendly drop-in replacement for `std::tuple` in gpu-array APIs.

Prefer `gpu_array::tuple` for tuple-like values that need to be used inside CUDA/HIP kernels. `std::tuple` can also be used with APIs such as `structure_of_arrays` when the backend standard library supports the required tuple operations in device code.

## `array` / `managed_array`

```cpp
template <typename T, typename size_type = size_type_default>
requires std::is_trivially_copyable_v<T>
class array;

template <typename T, typename size_type = size_type_default>
class managed_array;
```

The `array` and `managed_array` classes provide smart pointer-like wrappers for managing arrays on the GPU. They support C++20 ranges and iterator concepts, allowing seamless integration with modern C++ code and exporting to/from range-based containers. The managed variant uses unified memory for easy access from both host and device. The non-managed variant allocates memory on the device using `cudaMalloc/hipMalloc` and `cudaMemcpy/hipMemcpy` for data transfer, which requires the type `T` to be trivially copyable for safety.

Key semantics:

*   `array<T>` stores device-only memory. Host code can create, copy, reset, and convert it with `to<>()`, but direct element access through `operator[]`, iterators, `front`, `back`, or `data()` is intended for device code. Use `to<Container>()` to inspect values on the host.
*   `managed_array<T>` stores unified memory. Its elements can be accessed directly from both host and device code, and it can hold non-trivially-copyable C++ objects because elements are constructed and destroyed individually.
*   Copying an `array` or `managed_array` wrapper is shallow: the copy shares the same allocation and increments the internal use count. Use `to<array<U>>()` or `to<managed_array<U>>()` when an explicit copy of the data is required.
*   Empty arrays have size `0`, a null data pointer, and convert to `false` in boolean contexts.
*   Nested host ranges such as `std::vector<std::vector<T>>` are represented as nested `managed_array` values. Conceptually, class template argument deduction maps `std::vector<std::vector<int>>` to `managed_array<managed_array<int>>`; the exact size types follow `size_type_default` and the `GPU_USE_32BIT_SIZE_TYPE_DEFAULT` macro.

### Constructors

```cpp
// (1) default constructor
array();
managed_array();

// (2) copy/move constructors
__host__ __device__ array(const array& other);
__host__ __device__ array(array&& other) noexcept;
__host__ __device__ managed_array(const managed_array& other);
__host__ __device__ managed_array(managed_array&& other) noexcept;

// (3) construct with size
__host__ explicit array(std::size_t n);
__host__ array(std::size_t n, const T& init_value);
__host__ array(std::size_t n, default_init_tag default_init);
__host__ explicit managed_array(std::size_t n);
__host__ managed_array(std::size_t n, const T& init_value);
__host__ managed_array(std::size_t n, default_init_tag default_init);

// (4) construct from range (e.g., std::vector, std::array)
template <std::ranges::input_range Range>
__host__ explicit array(const Range& range);
template <std::ranges::input_range Range>
__host__ explicit managed_array(const Range& range);
__host__ array(std::initializer_list<T> list);
__host__ managed_array(std::initializer_list<T> list);

// (5) construct from raw pointer (device pointer)
__host__ array(T* device_ptr, std::size_t n);
__device__ array(T* device_ptr, size_type n);
__host__ managed_array(T* device_ptr, std::size_t n);
__device__ managed_array(T* device_ptr, size_type n);
```

Where:

1.  Default constructor creates an empty array with null pointer.
2.  Copy and move constructors copy the wrapper state, not the data. The resulting wrappers share the same allocation.
3.  Constructors with size allocate memory on the GPU. Without an explicit tag, elements are value-initialized. Passing an initial value copies that value into each element. Passing `default_init` performs default initialization instead.
4.  Constructors from ranges copy data from host containers to device memory. Copying from `array` and `managed_array` types is not allowed to avoid unintended device-to-device copies. Use `to<>()` method for explicit device-to-device copy instead.
5.  Constructors from raw device pointers wrap existing device memory.

For nested ranges, nested `managed_array` is deduced: `std::vector<std::vector<T>> -> managed_array<managed_array<T>>`.

For `array<T>`, `T` must be trivially copyable. For `managed_array<T>`, `T` does not need to be trivially copyable, but it must be usable from whichever execution space accesses it. If a `managed_array<T>` is accessed from device code, the operations used by the kernel must be device-callable.

### Exporters

```cpp
// (1) Copy data to host container
template <typename Container>
__host__ Container to() const;
template <template <typename...> typename Container>
__host__ Container<T> to() const;
template <template <typename...> typename Container>
__host__ Container<Container<...>> to() const; // nested ranges deduction only for managed_array

// (2) Copy data to gpu array
template <typename U>
__host__ array<U> to<array<U>>() const;
template <typename U>
__host__ managed_array<U> to<managed_array<U>>() const;

// (3) Static cast to host container
template <typename Container>
__host__ explicit operator Container() const;
```

Where:

1.  `to<Container>()` copies data from device to host container (e.g., `std::vector<T>`, `std::list<T>`). Range value type can be deduced automatically (e.g., `to<std::vector>()`). For fixed-size containers such as `std::array`, the requested size must match the array size. For nested ranges, nested containers are deduced only for `managed_array`, (e.g., `managed_array<managed_array<U>>::to<std::vector> -> std::vector<std::vector<U>>`).
2.  `to<array<U>>()` and `to<managed_array<U>>()` copy data from device array to another gpu-array array type with element type `U`. The element type may change if each value is convertible to `U`.
3.  Explicit conversion operator to host container, equivalent to `to<Container>()`, but conversion to gpu-array types are not supported.

### Range interface

Member types:

```cpp
array::size_type;
array::value_type;
array::reference;
array::const_reference;
array::iterator;
array::const_iterator;
array::pointer;
array::const_pointer;

managed_array::size_type;
managed_array::value_type;
managed_array::reference;
managed_array::const_reference;
managed_array::iterator;
managed_array::const_iterator;
managed_array::pointer;
managed_array::const_pointer;
```

Member functions:

```cpp
__host__ __device__ reference operator[](size_type index) noexcept;
__host__ __device__ const_reference operator[](size_type index) const noexcept;
__host__ __device__ iterator begin() noexcept;
__host__ __device__ const_iterator begin() const noexcept;
__host__ __device__ iterator end() noexcept;
__host__ __device__ const_iterator end() const noexcept;
__host__ __device__ const_iterator cbegin() const noexcept;
__host__ __device__ const_iterator cend() const noexcept;
__host__ __device__ std::reverse_iterator<iterator> rbegin() noexcept;
__host__ __device__ std::reverse_iterator<const_iterator> rbegin() const noexcept;
__host__ __device__ std::reverse_iterator<iterator> rend() noexcept;
__host__ __device__ std::reverse_iterator<const_iterator> rend() const noexcept;
__host__ __device__ reference front() noexcept;
__host__ __device__ const_reference front() const noexcept;
__host__ __device__ reference back() noexcept;
__host__ __device__ const_reference back() const noexcept;
__host__ __device__ pointer data() noexcept;
__host__ __device__ const_pointer data() const noexcept;
__host__ __device__ size_type size() const noexcept;
__host__ __device__ bool empty() const noexcept;
```

Concepts:

```cpp
std::ranges::range<array<T>>;
std::ranges::borrowed_range<array<T>>;  // only for device code
std::ranges::view<array<T>>;
std::ranges::output_range<array<T>, T>;
std::ranges::input_range<array<T>>;
std::ranges::forward_range<array<T>>;
std::ranges::bidirectional_range<array<T>>;
std::ranges::random_access_range<array<T>>;
std::ranges::sized_range<array<T>>;
std::ranges::contiguous_range<array<T>>;
std::ranges::common_range<array<T>>;
std::ranges::viewable_range<array<T>>;

std::ranges::range<managed_array<T>>;
std::ranges::borrowed_range<managed_array<T>>;  // only for device code
std::ranges::view<managed_array<T>>;
std::ranges::output_range<managed_array<T>, T>;
std::ranges::input_range<managed_array<T>>;
std::ranges::forward_range<managed_array<T>>;
std::ranges::bidirectional_range<managed_array<T>>;
std::ranges::random_access_range<managed_array<T>>;
std::ranges::sized_range<managed_array<T>>;
std::ranges::contiguous_range<managed_array<T>>;
std::ranges::common_range<managed_array<T>>;
std::ranges::viewable_range<managed_array<T>>;
```

`array` and `managed_array` model contiguous, sized, random-access ranges. They also model `std::ranges::view` because the wrapper is a lightweight handle to GPU memory. `std::ranges::borrowed_range` is enabled only for device code; host code should treat iterators from these wrappers as tied to the wrapper lifetime.

Host access rules:

*   `array<T>` element accessors and iterators expose device memory. They are available so the same wrapper can be used in kernels, but host code should use `to<>()` instead of dereferencing them.
*   `managed_array<T>` element accessors and iterators can be used directly on the host, including range-based for loops, reverse iterators, `front`, and `back`.

### Smart pointer interface

```cpp
// (1) Reset pointer and size
__host__ __device__ void reset();
__host__ void reset(T* device_ptr, std::size_t n);
__device__ void reset(T* device_ptr, size_type n);

// (2) Boolean conversion
__host__ __device__ explicit operator bool() const noexcept;

// (3) Use count
__host__ std::uint32_t use_count() const noexcept;
```

Where:

1.  If host code calls `reset(...)`, the current device memory is freed and set new device pointer and size. If device code calls `reset(...)`, it only sets the internal pointer and size without freeing memory.
2.  Bool conversion operator to check if the internal pointer is not null.
3.  Returns the current use count of the internal pointer. Note that this is only valid in host code.

Note: The device-side `reset` function does not affect to the memory management on the host side. It only changes the internal pointer and size on the device side.

Calling `reset()` on one shared wrapper releases that wrapper's ownership and makes it empty. Other wrappers that share the same allocation remain valid until their ownership is released. Raw-pointer constructors are intended for wrapping existing GPU allocations; the wrapper then participates in the same RAII/use-count machinery as other array objects.

### Memory management

Note: Memory management functions are only available for `managed_array` since they use unified memory.

```cpp
// (1) Prefetch
__host__ void prefetch(size_type start_idx, size_type len, int device_id, api::gpuStream_t stream = 0, bool recursive = true) const;
__host__ void prefetch(size_type start_idx, size_type len, api::gpuStream_t stream = 0, bool recursive = true) const;
__host__ void prefetch(int device_id, api::gpuStream_t stream = 0, bool recursive = true) const;
__host__ void prefetch(api::gpuStream_t stream = 0, bool recursive = true) const;

// (2) Prefetch to host memory
__host__ void prefetch_to_cpu(size_type start_idx, size_type len, api::gpuStream_t stream = 0, bool recursive = true) const;
__host__ void prefetch_to_cpu(api::gpuStream_t stream = 0, bool recursive = true) const;

// (3) Memory advice
__host__ void mem_advise(size_type n, size_type len, api::gpuMemoryAdvise advise, int device_id, bool recursive = true) const;
__host__ void mem_advise(size_type n, size_type len, api::gpuMemoryAdvise advise, bool recursive = true) const;
__host__ void mem_advise(api::gpuMemoryAdvise advise, int device_id, bool recursive = true) const;
__host__ void mem_advise(api::gpuMemoryAdvise advise, bool recursive = true) const;

// (4) Memory advice to host memory
__host__ void mem_advise_to_cpu(size_type n, size_type len, api::gpuMemoryAdvise advise, bool recursive = true) const;
__host__ void mem_advise_to_cpu(api::gpuMemoryAdvise advise, bool recursive = true) const;
```

Where:

1.  Wrapper for `cudaMemPrefetchAsync/hipMemPrefetchAsync` to prefetch unified memory to specified device. The former overload prefetches a memory range, while the latter overload prefetches the entire memory. Overloads without `device_id` use the current device. If `recursive` is true and the value type of the array has `prefetch(...)` function, prefetch is called recursively for nested or member arrays.
2.  Host memory prefetching overloads with similar behavior to (1).
3.  Wrapper for `cudaMemAdvise/hipMemAdvise` to set memory advice for unified memory. The former overload sets advice for a memory range, while the latter overload sets advice for the entire memory. Overloads without `device_id` use the current device. If `recursive` is true and the value type of the array has `mem_advise(...)` function, mem_advise is called recursively for nested or member arrays.
4.  Host memory advice overloads with similar behavior to (3).

## `value` / `managed_value`

```cpp
template <typename T>
requires std::is_trivially_copyable_v<T>
class value;

template <typename T>
class managed_value;
```

The `value` and `managed_value` classes provide smart pointer-like wrappers for managing single values on the GPU. The managed variant uses unified memory for easy access from both host and device. The non-managed variant allocates memory on the device using `cudaMalloc/hipMalloc` and `cudaMemcpy/hipMemcpy` for data transfer, which requires the type `T` to be trivially copyable for safety.

Key semantics:

*   `value<T>` stores device-only memory. Host code can read a copy of the stored value with `operator*()` or the host `operator->()` proxy, but host dereference does not provide mutable access to device memory.
*   `managed_value<T>` stores unified memory. Host and device code can directly dereference it and mutate the object through `operator*()` or `operator->()`.
*   Copying a `value` or `managed_value` wrapper is shallow: the copy shares the same allocation and increments the internal use count.
*   Empty values have a null pointer and convert to `false` in boolean contexts.
*   Raw-pointer constructors wrap an existing GPU allocation and transfer ownership to the wrapper's RAII/use-count machinery.

### Constructors

```cpp
// (1) default constructor
value();
managed_value();

// (2) copy/move constructors
__host__ __device__ value(const value& other);
__host__ __device__ value(value&& other) noexcept;
__host__ __device__ managed_value(const managed_value& other);
__host__ __device__ managed_value(managed_value&& other) noexcept;

// (3) construct with initial value
__host__ explicit value(const T& init_value);
__host__ explicit managed_value(const T& init_value);
__host__ explicit value(default_init_tag default_init);
__host__ explicit managed_value(default_init_tag default_init);

// (4) Construct the element in-place by arguments
template <typename... Args>
__host__ explicit value(Args&&... args);
template <typename... Args>
__host__ explicit managed_value(Args&&... args);

// (5) construct from raw pointer (device pointer)
__host__ __device__ value(T* device_ptr);
__host__ __device__ managed_value(T* device_ptr);
```

Where:

1.  Default constructor creates an empty value with null pointer.
2.  Copy and move constructors copy the wrapper state, not the data. The resulting wrappers share the same allocation.
3.  Constructors with initial value or [default initialization](https://en.cppreference.com/w/cpp/language/default_initialization.html). For trivial types, `default_init` leaves the stored value default-initialized rather than value-initialized.
4.  Constructors that forward arguments to construct the element in-place. The arguments are perfectly forwarded to the constructor of `T`.
5.  Constructors from raw device pointers wrap existing device memory. `value<T>` expects device memory and `managed_value<T>` expects managed memory.

For `value<T>`, `T` must be trivially copyable. For `managed_value<T>`, `T` does not need to be trivially copyable, but it must be usable from whichever execution space accesses it.

Note: The device-side `reset` function does not affect to the memory management on the host side. It only changes the internal pointer and size on the device side.

### Smart pointer interface

Member types:

```cpp
value::element_type;
managed_value::element_type;
```

Member functions:

```cpp
// (1) Operators for `value`
__device__ T& operator*() const noexcept;
__host__ T operator*() const;
__device__ T* operator->() const noexcept;
__host__ proxy_object operator->() const;

// (2) Operators for `managed_value`
__host__ __device__ T& operator*() const noexcept;
__host__ __device__ T* operator->() const noexcept;

// (3) Get raw pointer
__host__ __device__ T* get() const noexcept;

// (4) Reset pointer
__host__ __device__ void reset(T* device_ptr = nullptr);

// (5) Boolean conversion
__host__ __device__ explicit operator bool() const noexcept;

// (6) Use count
__host__ std::uint32_t use_count() const noexcept;
```

Where:

1.  Dereference and member access operators for `value`. In host code, `operator*()` returns a copy transferred from device memory, and `operator->()` returns a proxy object that points to a copy of the value. In device code, these operators access the device object directly.
2.  Dereference and member access operators for `managed_value`, available in both host and device code.
3.  Get the raw device pointer.
4.  If host code calls `reset(...)`, the current device memory is freed and set new device pointer. If device code calls `reset(...)`, it only sets the internal pointer without freeing memory.
5.  Bool conversion operator to check if the internal pointer is not null.
6.  Returns the current use count of the internal pointer. Note that this is only valid in host code.

Note: The device-side `reset` function does not affect to the memory management on the host side. It only changes the internal pointer on the device side.

### Memory management

Note: Memory management functions are only available for `managed_value` since they use unified memory.

```cpp
// (1) Prefetch
__host__ void prefetch(int device_id, api::gpuStream_t stream = 0, bool recursive = true) const;
__host__ void prefetch(api::gpuStream_t stream = 0, bool recursive = true) const;

// (2) Prefetch to host memory
__host__ void prefetch_to_cpu(api::gpuStream_t stream = 0, bool recursive = true) const;

// (3) Memory advice
__host__ void mem_advise(api::gpuMemoryAdvise advise, int device_id, bool recursive = true) const;
__host__ void mem_advise(api::gpuMemoryAdvise advise, bool recursive = true) const;

// (4) Memory advice to host memory
__host__ void mem_advise_to_cpu(api::gpuMemoryAdvise advise, bool recursive = true) const;
```

Where:

1.  Wrapper for `cudaMemPrefetchAsync/hipMemPrefetchAsync` to prefetch unified memory to specified device. The overload without `device_id` uses the current device. If `recursive` is true and the value type has `prefetch(int, api::gpuStream_t)` function, prefetch is called recursively for the stored object.
2.  Host memory prefetching overload with similar behavior to (1).
3.  Wrapper for `cudaMemAdvise/hipMemAdvise` to set memory advice for unified memory. The overload without `device_id` uses the current device. If `recursive` is true and the value type has `mem_advise(api::gpuMemoryAdvise, int)` function, mem_advise is called recursively for the stored object.
4.  Host memory advice overload with similar behavior to (3).

## `structure_of_arrays` / `managed_structure_of_arrays`

```cpp
template <typename... Ts>
class structure_of_arrays;  // value_type is gpu_array::tuple<Ts...>
template <template <typename...> typename Tuple, typename... Ts, typename SizeType = size_type_default>
class structure_of_arrays<Tuple<Ts...>, SizeType>;


template <typename... Ts>
class managed_structure_of_arrays;  // value_type is gpu_array::tuple<Ts...>
template <template <typename...> typename Tuple, typename... Ts, typename SizeType = size_type_default>
class managed_structure_of_arrays<Tuple<Ts...>, SizeType>;
```

The `structure_of_arrays` and `managed_structure_of_arrays` classes provide smart pointer-like wrappers for managing Structure-of-Arrays (SoA) memory layout on the GPU. They allow for optimized memory access patterns by storing each member of a structure in separate contiguous arrays. The index access interface allows retrieval of the entire structure at a given index. This class is useful for maximizing coalesced memory access on GPUs. These classes support C++20 ranges and iterator concepts.

The value type of `structure_of_arrays<Ts...>` is `gpu_array::tuple<Ts...>`. If a structure-like accessor interface is desired, the value type can also be `gpu_array::tuple<Ts...>`, `std::tuple<Ts...>`, or a tuple-like class template derived from one of them. The tuple-like type must be constructible as `Tuple<Ts...>`, `Tuple<Ts&...>`, and `Tuple<const Ts&...>`, and each element must be accessible with `get<N>(value)` found by unqualified lookup. The example definition of such tuple-derived class is as follows:

```cpp
template <typename... Ts>
requires (sizeof...(Ts) == 3)
struct CustomTuple : public gpu_array::tuple<Ts...>
{
    using gpu_array::tuple<Ts...>::tuple;
    using gpu_array::tuple<Ts...>::operator=;
    __host__ __device__ decltype(auto) get_a() { return gpu_array::get<0>(*this); }
    __host__ __device__ decltype(auto) get_b() { return gpu_array::get<1>(*this); }
    __host__ __device__ decltype(auto) get_c() { return gpu_array::get<2>(*this); }
};
```

The same pattern can be used with `std::tuple<Ts...>` and `std::get` in host/device environments where the standard library tuple implementation is available in device code.

The template parameters `Ts...` correspond to the member types of the tuple-derived class. All parameters must be value types (i.e., not reference types), since the members are stored in separate arrays and returns by tuple of reference types of each element when accessed.

### Constructors

```cpp
// (1) default constructor
structure_of_arrays();
managed_structure_of_arrays();

// (2) copy/move constructors
__host__ __device__ structure_of_arrays(const structure_of_arrays& other);
__host__ __device__ structure_of_arrays(structure_of_arrays&& other) noexcept;
__host__ __device__ managed_structure_of_arrays(const managed_structure_of_arrays& other);
__host__ __device__ managed_structure_of_arrays(managed_structure_of_arrays&& other) noexcept;

// (3) construct with size
__host__ explicit structure_of_arrays(std::size_t n);
__host__ structure_of_arrays(std::size_t n, const Tuple<Ts...>& init_value);
__host__ structure_of_arrays(std::size_t n, default_init_tag default_init);
__host__ explicit managed_structure_of_arrays(std::size_t n);
__host__ managed_structure_of_arrays(std::size_t n, const Tuple<Ts...>& init_value);
__host__ managed_structure_of_arrays(std::size_t n, default_init_tag default_init);

// (4) construct from range of tuple-derived class
template <std::ranges::forward_range Range>
requires std::ranges::sized_range<Range>
__host__ explicit structure_of_arrays(const Range& range);
template <std::ranges::forward_range Range>
requires std::ranges::sized_range<Range>
__host__ explicit managed_structure_of_arrays(const Range& range);
__host__ structure_of_arrays(std::initializer_list<Tuple<Ts...>> list);
__host__ managed_structure_of_arrays(std::initializer_list<Tuple<Ts...>> list);

// (5) construct from multiple ranges
template <std::ranges::forward_range... Ranges>
requires (std::ranges::sized_range<Ranges> && ...)
__host__ explicit structure_of_arrays(const Ranges& ranges...);
__host__ explicit structure_of_arrays(std::initializer_list<Ts>... lists);
template <std::ranges::forward_range... Ranges>
requires (std::ranges::sized_range<Ranges> && ...)
__host__ explicit managed_structure_of_arrays(const Ranges& ranges...);
__host__ explicit managed_structure_of_arrays(std::initializer_list<Ts>... lists);
```

Where:

1.  Default constructor creates an empty structure_of_arrays with null pointers.
2.  Copy and move constructors for copying pointers and size.
3.  Constructors with size allocate memory on the GPU. Optionally, an initial value or [default initialization](https://en.cppreference.com/w/cpp/language/default_initialization.html).
4.  Constructors from ranges of tuple-derived class copy data from host containers to device memory. The range must be a sized forward range because each tuple field is copied by a separate pass over the source range. Copying from `structure_of_arrays` is not allowed to avoid unintended device-to-device copies; copying from `managed_structure_of_arrays` is allowed because it is host-accessible unified memory.
5.  Constructors from multiple sized forward ranges copy data from each host container to corresponding member arrays on the device.

### Exporters

```cpp
// (1) Copy data to host container
template <typename Container>
__host__ Container to() const;
template <template <typename...> typename Container>
__host__ Container<Tuple<Ts...>> to() const;

// (2) Static cast to host container
template <typename Container>
__host__ explicit operator Container() const;
```

Where:

1.  `to<Container>()` copies data from device to host container (e.g., `std::vector<Tuple<Ts...>>`, `std::list<Tuple<Ts...>>`). Range value type can be deduced automatically (e.g., `to<std::vector>() -> std::vector<Tuple<Ts...>>`).
2.  Explicit conversion operator to host container, equivalent to `to<Container>()`.

### Range interface

Member types:

```cpp
structure_of_arrays::size_type;
template <std::size_t N>
structure_of_arrays::element_type;

managed_structure_of_arrays::size_type;
template <std::size_t N>
managed_structure_of_arrays::element_type;
```

Member functions:

Return type notation used below:

*   `value` means `Tuple<Ts...>`.
*   `reference` means `Tuple<Ts&...>`.
*   `const_reference` means `Tuple<const Ts&...>`.
*   `iterator` and `const_iterator` mean random-access iterators whose reference types are `reference` and `const_reference`.

```cpp
__host__ __device__ reference operator[](size_type index) &;
__host__ __device__ const_reference operator[](size_type index) const&;
__host__ __device__ value operator[](size_type index) &&;
__host__ __device__ iterator begin() noexcept;
__host__ __device__ const_iterator begin() const noexcept;
__host__ __device__ iterator end() noexcept;
__host__ __device__ const_iterator end() const noexcept;
template <std::size_t N>
__host__ __device__ element_type<N>* data() noexcept;
template <std::size_t N>
__host__ __device__ const element_type<N>* data() const noexcept;
__host__ __device__ size_type size() const noexcept;
__host__ __device__ bool empty() const noexcept;
```

Concepts:

```cpp
using soa_type = structure_of_arrays<Tuple<Ts...>>;
using managed_soa_type = managed_structure_of_arrays<Tuple<Ts...>>;

std::ranges::range<soa_type>;
std::ranges::borrowed_range<soa_type>;  // only for device code
std::ranges::view<soa_type>;
std::ranges::output_range<soa_type, T>; // since C++23
std::ranges::input_range<soa_type>;
std::ranges::forward_range<soa_type>;
std::ranges::bidirectional_range<soa_type>;
std::ranges::random_access_range<soa_type>;
std::ranges::sized_range<soa_type>;
std::ranges::common_range<soa_type>;
std::ranges::viewable_range<soa_type>;

std::ranges::range<managed_soa_type>;
std::ranges::borrowed_range<managed_soa_type>;  // only for device code
std::ranges::view<managed_soa_type>;
std::ranges::output_range<managed_soa_type, T>; // since C++23
std::ranges::input_range<managed_soa_type>;
std::ranges::forward_range<managed_soa_type>;
std::ranges::bidirectional_range<managed_soa_type>;
std::ranges::random_access_range<managed_soa_type>;
std::ranges::sized_range<managed_soa_type>;
std::ranges::common_range<managed_soa_type>;
std::ranges::viewable_range<managed_soa_type>;
```

Note: `std::tuple<Ts...>` and `gpu_array::tuple<Ts...>` already provide the common-reference machinery needed by the standard range concepts. When you define your own tuple-derived class and use it as the element type of `structure_of_arrays` or `managed_structure_of_arrays`, you may need to inherit the base tuple assignment operator and specialize `std::common_type` and `std::basic_common_reference`. This is required when the SoA range is checked against concepts such as `std::ranges::input_range`, `std::ranges::forward_range`, or `std::ranges::random_access_range`, and when it is passed to sized range views such as `views::enumerate` or `views::zip`.

The specializations are not needed for `array<CustomTuple<...>>`, `managed_array<CustomTuple<...>>`, or simple SoA kernels that only use `views::grid_thread_stride` without constraining the kernel parameter as a standard range. Inheriting assignment is especially important when algorithms assign `Tuple<Ts...>` values through `Tuple<Ts&...>` references. For example:

```cpp
template <typename... Ts>
struct CustomTuple : public gpu_array::tuple<Ts...>
{
    using gpu_array::tuple<Ts...>::tuple;
    using gpu_array::tuple<Ts...>::operator=;
};

template <class... TTypes, class... UTypes>
requires requires { typename CustomTuple<std::common_type_t<TTypes, UTypes>...>; }
struct std::common_type<CustomTuple<TTypes...>, CustomTuple<UTypes...>>
{
    using type = CustomTuple<std::common_type_t<TTypes, UTypes>...>;
};

template <class... TTypes, class... UTypes, template <class> class TQual, template <class> class UQual>
requires requires { typename CustomTuple<std::common_reference_t<TQual<TTypes>, UQual<UTypes>>...>; }
struct std::basic_common_reference<CustomTuple<TTypes...>, CustomTuple<UTypes...>, TQual, UQual>
{
    using type = CustomTuple<std::common_reference_t<TQual<TTypes>, UQual<UTypes>>...>;
};
```

### Smart pointer interface

```cpp
// (1) Reset pointer and size
__host__ void reset();
template <std::size_t N>
__device__ void reset(pointer<N> device_ptr);
template <std::size_t N>
__device__ void reset(const array<Ts[N]>& device_array);
template <std::size_t N>
__device__ void reset(const managed_array<Ts[N]>& device_array);

// (2) Boolean conversion
__host__ __device__ explicit operator bool() const noexcept;

// (3) Use count
__host__ std::uint32_t use_count() const noexcept;
```

Where:

1.  If host code calls `reset()`, the current device memory is freed and set new device pointers and size. If device code calls `reset<N>(...)`, it only sets the internal pointers of `N`-th array without freeing memory. The overloads with `array` and `managed_array` set the internal pointers and checking size consistency with `assert()` from the given device arrays.
2.  Bool conversion operator to check if the internal pointer is not null.
3.  Returns the current use count of the internal pointer. Note that this is only valid in host code.

Note: The device-side `reset` function does not affect to the memory management on the host side. It only changes the internal pointer on the device side.

### Memory management

Note: Memory management functions are only available for `managed_structure_of_arrays` since they use unified memory.

```cpp
// (1) Prefetch
__host__ void prefetch(size_type start_idx, size_type len, int device_id, api::gpuStream_t stream = 0, bool recursive = true) const;
__host__ void prefetch(size_type start_idx, size_type len, api::gpuStream_t stream = 0, bool recursive = true) const;
__host__ void prefetch(int device_id, api::gpuStream_t stream = 0, bool recursive = true) const;
__host__ void prefetch(api::gpuStream_t stream = 0, bool recursive = true) const;

// (2) Prefetch to host memory
__host__ void prefetch_to_cpu(size_type start_idx, size_type len, api::gpuStream_t stream = 0, bool recursive = true) const;
__host__ void prefetch_to_cpu(api::gpuStream_t stream = 0, bool recursive = true) const;

// (3) Memory advice
__host__ void mem_advise(size_type n, size_type len, api::gpuMemoryAdvise advise, int device_id, bool recursive = true) const;
__host__ void mem_advise(size_type n, size_type len, api::gpuMemoryAdvise advise, bool recursive = true) const;
__host__ void mem_advise(api::gpuMemoryAdvise advise, int device_id, bool recursive = true) const;
__host__ void mem_advise(api::gpuMemoryAdvise advise, bool recursive = true) const;

// (4) Memory advice to host memory
__host__ void mem_advise_to_cpu(size_type n, size_type len, api::gpuMemoryAdvise advise, bool recursive = true) const;
__host__ void mem_advise_to_cpu(api::gpuMemoryAdvise advise, bool recursive = true) const;
```

Where:

1.  Wrapper for `cudaMemPrefetchAsync/hipMemPrefetchAsync` to prefetch unified memory to specified device. The former overload prefetches a memory range, while the latter overload prefetches the entire memory. Overloads without `device_id` use the current device. If `recursive` is true and the value type of the array has `prefetch(...)` function, prefetch is called recursively for nested or member arrays.
2.  Host memory prefetching overloads with similar behavior to (1).
3.  Wrapper for `cudaMemAdvise/hipMemAdvise` to set memory advice for unified memory. The former overload sets advice for a memory range, while the latter overload sets advice for the entire memory. Overloads without `device_id` use the current device. If `recursive` is true and the value type of the array has `mem_advise(...)` function, mem_advise is called recursively for nested or member arrays.
4.  Host memory advice overloads with similar behavior to (3).

## `jagged_array`

```cpp
template <gpu_managed_random_access_range ArrayType>
class jagged_array : public ArrayType;
```

The `jagged_array` class provides wrapper for managing multi-dimensional arrays with varying row lengths (jagged arrays) on the GPU. It inherits from the base array type, which can be either `managed_array<T>` or `managed_structure_of_arrays<Tuple<Ts...>>`, to utilize their memory management and range interfaces. The jagged array has additional offsets to handle varying row sizes, allowing efficient access to elements using multi-dimensional indices.

Note that the only internal storage types currently supported are `managed_array` and `managed_structure_of_arrays`.

### Constructors

```cpp
// (1) default constructor
jagged_array();

// (2) construct from sizes
template <std::ranges::input_range SizeRange>
__host__ explicit jagged_array(const SizeRange& sizes);
__host__ explicit jagged_array(std::initializer_list<size_type> sizes);

// (3) construct from sizes and base array (for managed_array)
template <std::ranges::input_range SizeRange>
__host__ jagged_array(const SizeRange& sizes, const managed_array<T, SizeType>& base_array);
__host__ jagged_array(std::initializer_list<size_type> sizes, const managed_array<T, SizeType>& base_array);

// (4) construct from sizes and base array (for managed_structure_of_arrays)
template <std::ranges::input_range SizeRange>
__host__ jagged_array(const SizeRange& sizes, const managed_structure_of_arrays<Tuple<Ts...>, SizeType>& base_array);
__host__ jagged_array(std::initializer_list<size_type> sizes, const managed_structure_of_arrays<Tuple<Ts...>, SizeType>& base_array);

// (5) construct from sizes and flat host container
template <std::ranges::input_range SizeRange, std::ranges::input_range Container>
__host__ jagged_array(const SizeRange& sizes, const Container& range);
__host__ jagged_array(std::initializer_list<size_type> sizes, const Container& range);

// (6) construct from nested host container
template <std::ranges::input_range NestedContainer>
__host__ jagged_array(const NestedContainer& nested_range);
__host__ jagged_array(std::initializer_list<std::initializer_list<T>> nested_list); // for managed_array
__host__ jagged_array(std::initializer_list<std::initializer_list<Tuple<Ts...>>> nested_list); // for managed_structure_of_arrays
```

Where:

1.  Default constructor creates an empty jagged_array with null pointers.
2.  Constructors from sizes allocate memory on the GPU for the jagged array based on the provided row sizes. The sizes can be provided as a range or an initializer list.
3.  Constructors from sizes and base array for `managed_array` type. The base array should contain the concatenated elements of all rows. The data is not copied; it is shared with the provided base array.
4.  Constructors from sizes and base array for `managed_structure_of_arrays` type. The base array should contain the concatenated elements of all rows in SoA layout. The data is not copied; it is shared with the provided base array.
5.  Constructors from sizes and flat host container copy data from the provided host container to the jagged array on the device. The host container should contain the concatenated elements of all rows.
6.  Constructors from nested host container copy data from the provided nested host container to the each row of jagged array on the device.

For constructors that take row sizes and an existing flat sequence, the sum of the row sizes must match the flat sequence size; otherwise, `std::invalid_argument` is thrown. Empty rows are supported. A default-constructed `jagged_array` has `size() == 0` and `num_rows() == 0`.

### Exporters

Inherited from the base array type (`managed_array` or `managed_structure_of_arrays`).

### Range interface

Inherited from the base array type (`managed_array` or `managed_structure_of_arrays`).

Additional member functions:

```cpp
// (1) Range interface for each row
__host__ __device__ detail::subrange row(size_type i) noexcept;
__host__ __device__ detail::subrange row(size_type i) const noexcept;
__host__ __device__ auto begin(size_type i) noexcept;
__host__ __device__ auto begin(size_type i) const noexcept;
__host__ __device__ auto end(size_type i) noexcept;
__host__ __device__ auto end(size_type i) const noexcept;
__host__ __device__ auto data(size_type i) noexcept;        // if base is managed_array
__host__ __device__ auto data(size_type i) const noexcept;  // if base is managed_array
template <std::size_t N>
__host__ __device__ auto* data() noexcept;                  // if base is managed_structure_of_arrays
template <std::size_t N>
__host__ __device__ const auto* data() const noexcept;      // if base is managed_structure_of_arrays
template <std::size_t N>
__host__ __device__ auto* data(size_type i) noexcept;       // if base is managed_structure_of_arrays
template <std::size_t N>
__host__ __device__ const auto* data(size_type i) const noexcept; // if base is managed_structure_of_arrays
__host__ __device__ size_type size(size_type i) const noexcept;
__host__ __device__ size_type num_rows() const noexcept;

// (2) Indexing operator with multi-dimensional indices
__host__ __device__ decltype(auto) operator[](std::array<size_type, 2> idx) &;
__host__ __device__ decltype(auto) operator[](std::array<size_type, 2> idx) const&;
__host__ __device__ decltype(auto) operator[](std::array<size_type, 2> idx) &&;
__host__ __device__ decltype(auto) operator[](size_type i, size_type j) &;      // for C++23
__host__ __device__ decltype(auto) operator[](size_type i, size_type j) const&; // for C++23
__host__ __device__ decltype(auto) operator[](size_type i, size_type j) &&;     // for C++23
```

### Smart pointer interface

Inherited from the base array type (`managed_array` or `managed_structure_of_arrays`).

### Memory management

The `prefetch`, `prefetch_to_cpu`, `mem_advise`, and `mem_advise_to_cpu` member functions apply to both the inherited base storage and the internal row-offset storage. Recursive memory management is forwarded to element values when supported by the base storage type.

## Grid-stride range adapter

The following range adapter closure objects in the `views` namespace are provided for grid-stride loops in GPU kernels.

```cpp
views::block_thread_stride;
views::grid_thread_stride;
views::grid_block_stride;
views::cluster_thread_stride;   // [*]
views::cluster_block_stride;    // [*]
views::grid_cluster_stride;     // [*]

// [*] Available only on CUDA backend [CC 9.0](https://developer.nvidia.com/cuda/gpus) or above.
```

They produce views that enable advancing the N-th element of the original range by a specified stride M. The pairs N and M correspond to the index of the thread/block/cluster within the block/cluster/grid and the number of threads/blocks/clusters, respectively.

On the host side, these adapters fall back to a serial traversal with initial index 0 and stride 1. This makes the same range expressions usable in host-side tests, while the actual grid-stride distribution is applied only in GPU device code.

The view object stores the range handle by value. Iterators obtained from the view should not outlive the view object.

### `views::block_thread_stride`

The stride access based on the **thread index** and the number of threads in the **block**.

Example usage:

```cpp
for (auto& v : array | views::block_thread_stride)
{
    // loop body
    v = ...;
}
```

which is equivalent to:

```cpp
const auto block = cooperative_groups::this_thread_block();
for (auto i = static_cast<decltype(array.size())>(block.thread_rank()); i < array.size(); i += block.num_threads())
{
    // loop body
    array[i] = ...;
}
```

### `views::cluster_thread_stride`

The stride access based on the **thread index** and the number of threads in the **cluster**.

Example usage:

```cpp
for (auto& v : array | views::cluster_thread_stride)
{
    // loop body
    v = ...;
}
```

which is equivalent to:

```cpp
const auto cluster = cooperative_groups::this_cluster();
for (auto i = static_cast<decltype(array.size())>(cluster.thread_rank()); i < array.size(); i += cluster.num_threads())
{
    // loop body
    array[i] = ...;
}
```

### `views::grid_thread_stride`

The stride access based on the **thread index** and the number of threads in the **grid**.

Example usage:

```cpp
for (auto& v : array | views::grid_thread_stride)
{
    // loop body
    v = ...;
}
```

which is equivalent to:

```cpp
const auto grid = cooperative_groups::this_grid();
for (auto i = static_cast<decltype(array.size())>(grid.thread_rank()); i < array.size(); i += grid.num_threads())
{
    // loop body
    array[i] = ...;
}
```

### `views::cluster_block_stride`

The stride access based on the **block index** and the number of blocks in the **cluster**.

Example usage:

```cpp
for (auto& v : array | views::cluster_block_stride)
{
    // loop body
    v = ...;
}
```

which is equivalent to:

```cpp
const auto cluster = cooperative_groups::this_cluster();
for (auto i = static_cast<decltype(array.size())>(cluster.block_rank()); i < array.size(); i += cluster.num_blocks())
{
    // loop body
    array[i] = ...;
}
```

### `views::grid_block_stride`

The stride access based on the **block index** and the number of blocks in the **grid**.

Example usage:

```cpp
for (auto& v : array | views::grid_block_stride)
{
    // loop body
    v = ...;
}
```

which is equivalent to:

```cpp
const auto grid = cooperative_groups::this_grid();
for (auto i = static_cast<decltype(array.size())>(grid.block_rank()); i < array.size(); i += grid.num_blocks())
{
    // loop body
    array[i] = ...;
}
```

### `views::grid_cluster_stride`

The stride access based on the **cluster index** and the number of clusters in the **grid**.

Example usage:

```cpp
for (auto& v : array | views::grid_cluster_stride)
{
    // loop body
    v = ...;
}
```

which is equivalent to:

```cpp
const auto grid = cooperative_groups::this_grid();
for (auto i = static_cast<decltype(array.size())>(grid.cluster_rank()); i < array.size(); i += grid.num_clusters())
{
    // loop body
    array[i] = ...;
}
```

## Enumerate view

`views::enumerate` and `enumerate_view` iterate over a sized random-access range as pairs of `(index, value)`. The index starts at 0 and the value is a reference to the original element, so assignments through the structured binding update the underlying range.

```cpp
for (auto&& [i, v] : array | views::enumerate)
{
    v += static_cast<int>(i);
}
```

The element type is represented with `gpu_array::tuple`, so both structured bindings and `gpu_array::get<N>` are supported.

```cpp
auto item = (array | views::enumerate)[2];
auto i = gpu_array::get<0>(item);
auto& v = gpu_array::get<1>(item);
```

`views::enumerate` can be composed with stride adapters when enumeration comes first. For example, `array | views::enumerate | views::grid_thread_stride` distributes enumerated `(index, value)` pairs across threads while preserving indices from the original range. The reverse order, such as `array | views::grid_thread_stride | views::enumerate`, is not supported because stride views do not model `std::ranges::sized_range`.

## Zip view

`views::zip` and `zip_view` iterate over multiple sized random-access ranges in lock step. The resulting view length is the minimum size of the input ranges; extra elements in longer ranges are not visited.

```cpp
for (auto&& [x, y] : views::zip(lhs, rhs))
{
    x += y;
}
```

The element type is represented with `gpu_array::tuple` containing references to the original ranges. `views::zip` is a function-style adapter; construct the zip first, then compose the resulting view with other adapters when needed.

```cpp
for (auto&& [i, pair] : views::zip(lhs, rhs) | views::enumerate | views::grid_thread_stride)
{
    auto&& [x, y] = pair;
    x += y * static_cast<int>(i + 1);
}
```

Composition order controls the element shape. `views::zip(lhs, rhs) | views::enumerate` yields `(index, (lhs_value, rhs_value))`, while `views::zip(lhs | views::enumerate, rhs)` yields `((index, lhs_value), rhs_value)`. Apply stride adapters after `views::enumerate` or `views::zip`; strided views are intentionally not sized ranges, so they cannot be passed to `views::enumerate` or `views::zip`.

## Utilities

### CUDA/HIP API wrappers

The `gpu_array::api` namespace provides wrappers for commonly used CUDA and HIP API functions and types. The API functions are prefixed with `gpu` to avoid name conflicts instead of `cuda` or `hip`. See the definitions in the [gpu_runtime_api.hpp](../include/gpu_runtime_api.hpp) file for details.

### Macros

**Backend selection:**

Define `ENABLE_HIP` to use HIP backend. Otherwise, CUDA backend is used by default.

**Default size type selection:**

Define `GPU_USE_32BIT_SIZE_TYPE_DEFAULT` to use `std::uint32_t` as the default size type for array-like classes. Otherwise, `std::size_t` is used by default.

**API error checking:**

`GPU_CHECK_ERROR()` function macro to check CUDA/HIP API errors. If an error occurs, it throws a `std::runtime_error` with the error message. Example usage:

```cpp
GPU_CHECK_ERROR(gpu_array::api::gpuGetDevice(&device_id));
```

**Device and host compilation macros:**

gpu-array library defines `GPU_DEVICE_COMPILE`, `GPU_OVERLOAD_DEVICE`, and `GPU_OVERLOAD_HOST` macros depending on host or device code compilation. The `GPU_DEVICE_COMPILE` macro is defined when compiling device code. The `GPU_OVERLOAD_DEVICE` and `GPU_OVERLOAD_HOST` macros handle the differences in behavior between CUDA and HIP for [overloading based on host and device code](https://llvm.org/docs/CompileCudaWithLLVM.html#overloading-based-on-host-and-device-attributes). The nvcc does not allow overloading based on `__host__` and `__device__` attributes with the same function signature, while hipcc allows it.

Example usage:

```cpp
__host__ __device__ void func()
{
#ifdef GPU_DEVICE_COMPILE
    // Device code
#else
    // Host code
#endif
}

#ifdef GPU_OVERLOAD_HOST
__host__ void foo()
{
    // Host code
}
#endif
#ifdef GPU_OVERLOAD_DEVICE
__device__ int foo()
{
    // Device code
}
#endif
```

