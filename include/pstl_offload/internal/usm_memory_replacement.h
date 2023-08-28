// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_PSTL_OFFLOAD_INTERNAL_USM_MEMORY_REPLACEMENT_H
#define _ONEDPL_PSTL_OFFLOAD_INTERNAL_USM_MEMORY_REPLACEMENT_H

#if !__SYCL_PSTL_OFFLOAD__
#    error "PSTL offload compiler mode should be enabled to use this header"
#endif

#include <atomic>
#include <cstdlib>
#include <cassert>
#include <cerrno>
#include <optional>

#include <sycl/sycl.hpp>

#include <oneapi/dpl/execution>

#include "usm_memory_replacement_common.h"

#if _WIN64
#    pragma comment(lib, "pstloffload.lib")
#endif

namespace __pstl_offload
{

static std::atomic<sycl::device*> __active_device = nullptr;

static void
__set_active_device(sycl::device* __new_active_device)
{
    __active_device.store(__new_active_device, std::memory_order_release);
}

static auto
__get_offload_device_selector()
{
#if __SYCL_PSTL_OFFLOAD__ == 1
    return sycl::default_selector_v;
#elif __SYCL_PSTL_OFFLOAD__ == 2
    return sycl::cpu_selector_v;
#elif __SYCL_PSTL_OFFLOAD__ == 3
    return sycl::gpu_selector_v;
#else
#    error "PSTL offload is not enabled or the selected value is unsupported"
#endif
}

class __offload_policy_holder_type
{
    using __set_active_device_func_type = void (*)(sycl::device*);

  public:
    // Since the global object of __offload_policy_holder_type is static but the constructor
    // of the class is inline, we need to avoid calling static functions inside of the constructor
    // and pass the pointer to exact function as an argument to guarantee that the correct __active_device
    // would be stored in each translation unit
    template <typename _DeviceSelector>
    __offload_policy_holder_type(const _DeviceSelector& __device_selector,
                                 __set_active_device_func_type __set_active_device_func)
        : _M_set_active_device(__set_active_device_func)
    {
        try
        {
            _M_offload_device.emplace(__device_selector);
            _M_set_active_device(&*_M_offload_device);
            _M_offload_policy.emplace(*_M_offload_device);
        }
        catch (const sycl::exception& e)
        {
            // __device_selector throws with e.code() == sycl::errc::runtime when device selection unable
            // to get offload device with required type. Do not pass an exception, as ctor is called for
            // a static object and the exception can't be processed.
            // Remember the situation and re-throw exception when asked for the policy from user's code.
            // Re-throw in every other case, as we don't know the reason.
            if (e.code() != sycl::errc::runtime)
                throw;
        }
    }

    ~__offload_policy_holder_type()
    {
        if (_M_offload_device.has_value())
            _M_set_active_device(nullptr);
    }

    auto
    __get_policy()
    {
        if (!_M_offload_device.has_value())
            throw sycl::exception(sycl::errc::runtime);
        return *_M_offload_policy;
    }
  private:
    std::optional<sycl::device> _M_offload_device;
    std::optional<oneapi::dpl::execution::device_policy<>> _M_offload_policy;
    __set_active_device_func_type _M_set_active_device;
}; // class __offload_policy_holder_type

static __offload_policy_holder_type __offload_policy_holder{__get_offload_device_selector(), __set_active_device};

#if __linux__
inline void*
__original_aligned_alloc(std::size_t __alignment, std::size_t __size)
{
    static __aligned_alloc_func_type __orig_aligned_alloc =
        __aligned_alloc_func_type(dlsym(RTLD_NEXT, "aligned_alloc"));
    return __orig_aligned_alloc(__alignment, __size);
}
#endif // __linux__

static void*
__internal_aligned_alloc(std::size_t __size, std::size_t __alignment)
{
    sycl::device* __device = __active_device.load(std::memory_order_acquire);
    void* __res = nullptr;

    if (__device != nullptr)
    {
        __res = __allocate_shared_for_device(__device, __size,
            __alignment? __alignment : alignof(std::max_align_t));
    }
    else
    {
        // note size/alignment args order for aligned allocation between Windows/Linux
#if _WIN64
        // Under Windows, memory with extended alignment must not be released by free() function,
        // so have to use malloc() for non-extended alignment allocations.
        __res = __alignment? __original_aligned_alloc(__alignment, __size) : __get_original_malloc()(__size);
#else
        __res = __original_aligned_alloc(__size, __alignment? __alignment : alignof(std::max_align_t));
#endif
    }

    if (__res && __alignment)
        assert((std::uintptr_t(__res) & (__alignment - 1)) == 0);
    return __res;
}

// This function is called by C allocation functions (malloc, calloc, etc)
// and sets errno on failure consistently with original memory allocating behavior
static void*
__errno_handling_internal_aligned_alloc(std::size_t __size, std::size_t __alignment)
{
    void* __ptr = __internal_aligned_alloc(__size, __alignment);
    if (__ptr == nullptr)
    {
        errno = ENOMEM;
    }
    return __ptr;
}

static void*
__internal_operator_new(std::size_t __size, std::size_t __alignment, bool __ext_alignment)
{
    // According to C++ standart "an alignment is ... the number of bytes between successive
    // addresses at which a given object can be allocated", so zero alignment is invalid.
    // Zero as __alignment value means that malloc (not aligned allocation) must be called
    // in nested calls, this distiction is vital for Windows.
    if (__ext_alignment && !__alignment)
    {
        throw std::bad_alloc{};
    }

    void* __res = __internal_aligned_alloc(__size, __alignment);

    while (__res == nullptr)
    {
        std::new_handler __handler = std::get_new_handler();
        if (__handler != nullptr)
        {
            __handler();
        }
        else
        {
            throw std::bad_alloc{};
        }
        __res = __internal_aligned_alloc(__size, __alignment);
    }

    return __res;
}

static void*
__internal_operator_new(std::size_t __size, std::size_t __alignment, bool __ext_alignment, const std::nothrow_t&) noexcept
{
    void* __res = nullptr;
    try
    {
        __res = __internal_operator_new(__size, __alignment, __ext_alignment);
    }
    catch (...)
    {
    }
    return __res;
}

} // namespace __pstl_offload

extern "C"
{

inline void* __attribute__((always_inline)) malloc(std::size_t __size)
{
    return ::__pstl_offload::__errno_handling_internal_aligned_alloc(__size, 0);
}

inline void* __attribute__((always_inline)) calloc(std::size_t __num, std::size_t __size)
{
    void* __res = nullptr;

    // Square root of maximal std::size_t value, values that are less never results in overflow during multiplication
    constexpr std::size_t __min_overflow_multiplier = std::size_t(1) << (sizeof(std::size_t) * CHAR_BIT / 2);
    std::size_t __allocate_size = __num * __size;

    // Check overflow on multiplication
    if ((__num >= __min_overflow_multiplier || __size >= __min_overflow_multiplier) &&
        (__num != 0 && __allocate_size / __num != __size))
    {
        errno = ENOMEM;
    }
    else
    {
        __res = ::__pstl_offload::__errno_handling_internal_aligned_alloc(__allocate_size, 0);
    }

    return __res ? std::memset(__res, 0, __allocate_size) : nullptr;
}

inline void* __attribute__((always_inline)) realloc(void* __ptr, std::size_t __size)
{
    return ::__pstl_offload::__internal_realloc(__ptr, __size);
}

#if __linux__

// valloc, pvalloc, __libc_valloc and __libc_pvalloc are not supported
// due to unsupported alignment on memory page

inline void* __attribute__((always_inline)) memalign(std::size_t __alignment, std::size_t __size) noexcept
{
    return ::__pstl_offload::__errno_handling_internal_aligned_alloc(__size, __alignment);
}

inline int __attribute__((always_inline)) posix_memalign(void** __memptr, std::size_t __alignment, std::size_t __size) noexcept
{
    int __result = 0;
    if (::__pstl_offload::__is_power_of_two(__alignment))
    {
        void* __ptr = ::__pstl_offload::__internal_aligned_alloc(__size, __alignment);

        if (__ptr != nullptr)
        {
            *__memptr = __ptr;
        }
        else
        {
            __result = ENOMEM;
        }
    }
    else
    {
        __result = EINVAL;
    }
    return __result;
}

inline int __attribute__((always_inline)) mallopt(int /*param*/, int /*value*/) noexcept { return 1; }

inline void* __attribute__((always_inline)) aligned_alloc(std::size_t __alignment, std::size_t __size)
{
    return ::__pstl_offload::__errno_handling_internal_aligned_alloc(__size, __alignment);
}

inline void* __attribute__((always_inline)) __libc_malloc(std::size_t __size)
{
    return malloc(__size);
}

inline void* __attribute__((always_inline)) __libc_calloc(std::size_t __num, std::size_t __size)
{
    return calloc(__num, __size);
}

inline void* __attribute__((always_inline)) __libc_memalign(std::size_t __alignment, std::size_t __size)
{
    return memalign(__alignment, __size);
}

inline void* __attribute__((always_inline)) __libc_realloc(void *__ptr, std::size_t __size)
{
    return realloc(__ptr, __size);
}

#elif _WIN64

inline void* __attribute__((always_inline)) _aligned_malloc(std::size_t __size, std::size_t __alignment)
{
    // _aligned_malloc should reject zero or not power of two alignments
    if (!::__pstl_offload::__is_power_of_two(__alignment))
    {
        errno = EINVAL;
        return nullptr;
    }
    return ::__pstl_offload::__errno_handling_internal_aligned_alloc(__size, __alignment);
}

inline void* __attribute__((always_inline)) _aligned_realloc(void* __ptr, std::size_t __size, std::size_t __alignment)
{
    // _aligned_realloc should reject zero or not power of two alignments
    if (!::__pstl_offload::__is_power_of_two(__alignment))
    {
        errno = EINVAL;
        return nullptr;
    }
    return ::__pstl_offload::__internal_aligned_realloc(__ptr, __size, __alignment);
}

#endif

} // extern "C"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winline-new-delete"

inline void* __attribute__((always_inline))
operator new(std::size_t __size)
{
    return ::__pstl_offload::__internal_operator_new(__size, 0, /*__alignment=*/false);
}

inline void* __attribute__((always_inline))
operator new[](std::size_t __size)
{
    return ::__pstl_offload::__internal_operator_new(__size, 0, /*__alignment=*/false);
}

inline void* __attribute__((always_inline))
operator new(std::size_t __size, const std::nothrow_t&) noexcept
{
    return ::__pstl_offload::__internal_operator_new(__size, 0, /*__alignment=*/false, std::nothrow);
}

inline void* __attribute__((always_inline))
operator new[](std::size_t __size, const std::nothrow_t&) noexcept
{
    return ::__pstl_offload::__internal_operator_new(__size, 0, /*__alignment=*/false, std::nothrow);
}

inline void* __attribute__((always_inline))
operator new(std::size_t __size, std::align_val_t __al)
{
    return ::__pstl_offload::__internal_operator_new(__size, std::size_t(__al), /*__alignment=*/true);
}

inline void* __attribute__((always_inline))
operator new[](std::size_t __size, std::align_val_t __al)
{
    return ::__pstl_offload::__internal_operator_new(__size, std::size_t(__al), /*__alignment=*/true);
}

inline void* __attribute__((always_inline))
operator new(std::size_t __size, std::align_val_t __al, const std::nothrow_t&) noexcept
{
    return ::__pstl_offload::__internal_operator_new(__size, std::size_t(__al), /*__alignment=*/true, std::nothrow);
}

inline void* __attribute__((always_inline))
operator new[](std::size_t __size, std::align_val_t __al, const std::nothrow_t&) noexcept
{
    return ::__pstl_offload::__internal_operator_new(__size, std::size_t(__al), /*__alignment=*/true, std::nothrow);
}

#pragma GCC diagnostic pop

#endif // _ONEDPL_PSTL_OFFLOAD_INTERNAL_USM_MEMORY_REPLACEMENT_H
