#pragma once

#include <kernel/program/compute/binding/model.hpp>

#include <memory>

namespace rund::node::accel::detail {

struct SolveBinds {
  const rund::kernel::ResidentBufferRef *primary = nullptr;
  const std::shared_ptr<void> *primary_handle = nullptr;
  const rund::kernel::ResidentBufferRef *aux = nullptr;
  const std::shared_ptr<void> *aux_handle = nullptr;
  const rund::kernel::ResidentBufferRef *rhs = nullptr;
  const std::shared_ptr<void> *rhs_handle = nullptr;
  const rund::kernel::ResidentBufferRef *output = nullptr;
  const std::shared_ptr<void> *output_handle = nullptr;
  const rund::kernel::ResidentBufferRef *status = nullptr;
  const std::shared_ptr<void> *status_handle = nullptr;
};

} // namespace rund::node::accel::detail
