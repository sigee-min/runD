#pragma once

#include <kernel/program/compute/binding/model.hpp>

#include <memory>

namespace rund::node::accel::detail {

struct FactorBinds {
  const rund::kernel::ResidentBufferRef *input = nullptr;
  const std::shared_ptr<void> *input_handle = nullptr;
  const rund::kernel::ResidentBufferRef *factor = nullptr;
  const std::shared_ptr<void> *factor_handle = nullptr;
  const rund::kernel::ResidentBufferRef *aux = nullptr;
  const std::shared_ptr<void> *aux_handle = nullptr;
  const rund::kernel::ResidentBufferRef *status = nullptr;
  const std::shared_ptr<void> *status_handle = nullptr;
};

} // namespace rund::node::accel::detail
