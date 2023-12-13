// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_PSTL_OFFLOAD_INTERNAL_USM_MEMORY_REPLACEMENT_COMMON_H
#define _ONEDPL_PSTL_OFFLOAD_INTERNAL_USM_MEMORY_REPLACEMENT_COMMON_H

#include <sycl/sycl.hpp>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <limits>

#if __linux__
#include <dlfcn.h>
#include <unistd.h>
#endif // __linux__

namespace __pstl_offload
{

constexpr bool
__is_power_of_two(std::size_t __number)
{
    return (__number != 0) && ((__number & __number - 1) == 0);
}

// can't use std::shared_ptr, because sizeof(std::shared_ptr) is 2*sizeof(void*) while
// sizeof(__block_header) must be power of 2 and with std::shared_ptr instead of
// __sycl_device_shared_ptr memory fragmentation increases drastically
class __sycl_device_shared_ptr
{
    struct __shared_device
    {
        std::optional<sycl::device> _M_device;
        // to keep reference to default context of the device as long as allocated memory objects exist
        std::optional<sycl::context> _M_default_context;
        std::atomic<std::size_t> _M_cnt;
    };

    __shared_device* _M_shared_device;

  public:
    template <typename _DeviceSelector>
    __sycl_device_shared_ptr(const _DeviceSelector& __device_selector)
        // new always allocates system memory at this point
        : _M_shared_device(new __shared_device{std::nullopt, std::nullopt, 1})
    {
        try
        {
            _M_shared_device->_M_device.emplace(__device_selector);
        }
        catch (const sycl::exception& e)
        {
            // __device_selector call throws with e.code() == sycl::errc::runtime when device selection unable
            // to get offload device with required type. Do not pass an exception, as ctor is called for
            // a static object and the exception can't be processed.
            // Remember the situation as empty _M_device and re-throw exception when asked for
            // the policy from user's code.
            // Re-throw in every other case, as we don't know the reason of an exception.
            if (e.code() == sycl::errc::runtime)
            {
                return;
            }
            else
            {
                throw;
            }
        }
        _M_shared_device->_M_default_context.emplace(
            _M_shared_device->_M_device->get_platform().ext_oneapi_get_default_context());
    }

    bool
    __is_device_created() const
    {
        return _M_shared_device->_M_device.has_value();
    }

    sycl::device
    __get_device() const
    {
        return *_M_shared_device->_M_device;
    }

    sycl::context
    __get_context() const
    {
        return *_M_shared_device->_M_default_context;
    }

    __sycl_device_shared_ptr&
    operator=(const __sycl_device_shared_ptr& other)
    {
        if (this != &other)
        {
            _M_shared_device = other._M_shared_device;
            ++_M_shared_device->_M_cnt;
        }
        return *this;
    }

    __sycl_device_shared_ptr(const __sycl_device_shared_ptr& other)
    {
        _M_shared_device = other._M_shared_device;
        ++_M_shared_device->_M_cnt;
    }

    ~__sycl_device_shared_ptr()
    {
        if (0 == --_M_shared_device->_M_cnt)
        {
            delete _M_shared_device;
        }
    }
};

inline constexpr std::size_t __uniq_type_const = 0x23499abc405a9bccLLU;

struct __block_header
{
    std::size_t _M_uniq_const;
    void* _M_original_pointer;
    __sycl_device_shared_ptr _M_device;
    std::size_t _M_requested_number_of_bytes;
}; // struct __block_header

static_assert(__is_power_of_two(sizeof(__block_header)));

void*
__allocate_shared_for_device_large_alignment(__sycl_device_shared_ptr __device_ptr, std::size_t __size, std::size_t __alignment);

void*
__realloc_impl(void* __user_ptr, std::size_t __new_size);

#if __linux__

inline std::size_t
__get_memory_page_size()
{
    static std::size_t __memory_page_size = sysconf(_SC_PAGESIZE);
    assert(__is_power_of_two(__memory_page_size));
    return __memory_page_size;
}

inline bool
__same_memory_page(void* __ptr1, void* __ptr2)
{
    std::uintptr_t __page_size = __get_memory_page_size();
    return (std::uintptr_t(__ptr1) ^ std::uintptr_t(__ptr2)) < __page_size;
}

inline void*
__allocate_shared_for_device(__sycl_device_shared_ptr __device_ptr, std::size_t __size, std::size_t __alignment)
{
    assert(__device_ptr.__is_device_created());
    // Impossible to guarantee that the returned pointer and memory header would be on the same memory
    // page if the alignment for more than a memory page is requested, so process this case specifically
    if (__alignment >= __get_memory_page_size())
    {
        return __allocate_shared_for_device_large_alignment(__device_ptr, __size, __alignment);
    }

    std::size_t __base_offset = std::max(__alignment, sizeof(__block_header));

    // Check overflow on addition of __base_offset and __size
    if (std::numeric_limits<std::size_t>::max() - __base_offset < __size)
    {
        return nullptr;
    }

    // Memory block allocated with sycl::aligned_alloc_shared should be aligned to at least sizeof(__block_header) * 2
    // to guarantee that header and header + sizeof(__block_header) (user pointer) would be placed in one memory page
    std::size_t __usm_alignment = __base_offset << 1;
    // Required number of bytes to store memory header and preserve alignment on returned pointer
    // usm_alignment bytes are reserved to store memory header
    std::size_t __usm_size = __size + __base_offset;

    sycl::device __device = __device_ptr.__get_device();
    sycl::context __context = __device_ptr.__get_context();
    void* __ptr = sycl::aligned_alloc_shared(__usm_alignment, __usm_size, __device, __context);

    if (__ptr != nullptr)
    {
        void* __original_pointer = __ptr;
        __ptr = static_cast<char*>(__ptr) + __base_offset;
        __block_header* __header = static_cast<__block_header*>(__ptr) - 1;
        assert(__same_memory_page(__ptr, __header));
        new (__header) __block_header{__uniq_type_const, __original_pointer, __device_ptr, __size};
    }

    return __ptr;
}

static void*
__internal_realloc(void* __user_ptr, std::size_t __new_size)
{
    // std::malloc() might be overloaded in per-TU overload, so keep it here
    return __user_ptr == nullptr ? std::malloc(__new_size) : __realloc_impl(__user_ptr, __new_size);
}

#endif // __linux__

} // namespace __pstl_offload

#endif // _ONEDPL_PSTL_OFFLOAD_INTERNAL_USM_MEMORY_REPLACEMENT_COMMON_H
