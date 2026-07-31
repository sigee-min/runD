#pragma once

#include <kernel/program/compute/binding/model.hpp>

#include <memory>

namespace rund::node::accel::detail {

struct PartitionBinds {
  const rund::kernel::ResidentBufferRef *flags = nullptr;
  const std::shared_ptr<void> *flags_handle = nullptr;
  const rund::kernel::ResidentBufferRef *values = nullptr;
  const std::shared_ptr<void> *values_handle = nullptr;
  const rund::kernel::ResidentBufferRef *output = nullptr;
  const std::shared_ptr<void> *output_handle = nullptr;
};

} // namespace rund::node::accel::detail
