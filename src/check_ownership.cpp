// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <excpt.h>

#include "pstl_offload_internal.h"

namespace __pstl_offload
{

// have to keep in it separate TU, because sycl is incompatible with Structured Exception Handling
bool
__is_our_memory(void* __user_ptr)
{
    bool __our;

    __try
    {
        // to generate a code for invalid access protection, __check_ownership_unsafe must not be inlined
        __our = __check_ownership_unsafe(__user_ptr);
    }
    __except (GetExceptionCode() == STATUS_ACCESS_VIOLATION ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH)
    {
        __our = false;
    }
    return __our;
}

}
