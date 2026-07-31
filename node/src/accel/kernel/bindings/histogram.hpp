#pragma once

#include <kernel/program/compute/binding/model.hpp>

#include <memory>

namespace rund::node::accel::detail {

struct HistogramBinds {
  const rund::kernel::ResidentBufferRef *bins = nullptr;
  const std::shared_ptr<void> *bins_handle = nullptr;
  const rund::kernel::ResidentBufferRef *counts = nullptr;
  const std::shared_ptr<void> *counts_handle = nullptr;
};

} // namespace rund::node::accel::detail
