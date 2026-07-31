#pragma once

#include <kernel/program/compute/binding/model.hpp>

#include <memory>

namespace rund::node::accel::detail {

struct SegmentedScanBinds {
  const rund::kernel::ResidentBufferRef *input = nullptr;
  const std::shared_ptr<void> *input_handle = nullptr;
  const rund::kernel::ResidentBufferRef *heads = nullptr;
  const std::shared_ptr<void> *heads_handle = nullptr;
  const rund::kernel::ResidentBufferRef *output = nullptr;
  const std::shared_ptr<void> *output_handle = nullptr;
};

struct SegmentedReduceBinds {
  const rund::kernel::ResidentBufferRef *input = nullptr;
  const std::shared_ptr<void> *input_handle = nullptr;
  const rund::kernel::ResidentBufferRef *heads = nullptr;
  const std::shared_ptr<void> *heads_handle = nullptr;
  const rund::kernel::ResidentBufferRef *output = nullptr;
  const std::shared_ptr<void> *output_handle = nullptr;
};

} // namespace rund::node::accel::detail
