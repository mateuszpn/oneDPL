// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _PSTL_OFFLOAD_INTERNAL_H
#define _PSTL_OFFLOAD_INTERNAL_H

namespace __pstl_offload
{

__declspec(noinline) bool
__check_ownership_unsafe(void* __user_ptr);

}

#endif // _PSTL_OFFLOAD_INTERNAL_H
