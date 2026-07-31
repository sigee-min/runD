#pragma once

#include <kernel/program/compute/binding/model.hpp>

#include <memory>

namespace rund::node::accel::detail {

struct SortBinds {
  const rund::kernel::ResidentBufferRef *read_keys = nullptr;
  const std::shared_ptr<void> *read_keys_handle = nullptr;
  const rund::kernel::ResidentBufferRef *read_values = nullptr;
  const std::shared_ptr<void> *read_values_handle = nullptr;
  const rund::kernel::ResidentBufferRef *write_keys = nullptr;
  const std::shared_ptr<void> *write_keys_handle = nullptr;
  const rund::kernel::ResidentBufferRef *write_values = nullptr;
  const std::shared_ptr<void> *write_values_handle = nullptr;
  const rund::kernel::ResidentBufferRef *logical_count = nullptr;
  const std::shared_ptr<void> *logical_count_handle = nullptr;
};

} // namespace rund::node::accel::detail
