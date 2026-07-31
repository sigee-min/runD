#pragma once

#include <rund/compute/abi/model.hpp>
#include <rund/compute/ops.hpp>

#include <cstdint>

namespace rund::compute::detail {

struct ScanStep final {
  std::uint32_t input{};
  std::uint32_t output{};
  std::uint32_t count{};
  Scan operation{Scan::InclusiveSum};
  FlowControl control{};
};

} // namespace rund::compute::detail
