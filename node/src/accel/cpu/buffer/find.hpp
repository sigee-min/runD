#pragma once

#include <accel/check.hpp>

#include "../../resident/ref.hpp"
#include "../../resident/result.hpp"
#include "../../resident/validation.hpp"
#include "../buffer.hpp"

#include <memory>
#include <utility>

namespace rund::node::accel::detail {

[[nodiscard]] inline CpuBufferResult
CpuResidentResult(std::shared_ptr<void> owner) {
  auto buffer = std::static_pointer_cast<CpuBuffer>(std::move(owner));
  return CpuBufferResult{
      .check = rund::AccelCheck{true, "ok"},
      .ref = RefFromResident(*buffer),
      .buffer = std::move(buffer),
  };
}

} // namespace rund::node::accel::detail
