#pragma once

#include <kernel/core/model.hpp>

namespace rund::kernel {

enum class WorkerAffinityPolicy : u8 {
  Static = 0,
};

enum class WorkerTruthLevel : u8 {
  Unknown = 0,
  HintOnly = 1,
  Verified = 2,
};

} // namespace rund::kernel
