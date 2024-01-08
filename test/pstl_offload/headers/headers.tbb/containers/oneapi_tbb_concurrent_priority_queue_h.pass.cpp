// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/tbb/concurrent_priority_queue.h>
#include "support/utils.h"

int main() {
    [[maybe_unused]] oneapi::tbb::concurrent_priority_queue<int> cpq;
    return TestUtils::done();
}
