#include "../receipts.hpp"

#include <utility>

namespace rund::compute::detail::graph_reduce {

CpuGraphQuarantine::CpuGraphQuarantine(
    std::shared_ptr<CpuReceiptBook> value) noexcept
    : book(std::move(value)) {}

CpuGraphQuarantine::~CpuGraphQuarantine() noexcept = default;

bool CpuGraphQuarantine::empty() const noexcept {
  return phase == Phase::Ready && self == nullptr &&
         (book == nullptr || book->idle());
}

bool CpuGraphQuarantine::bind(
    const PipelineState *const pipeline_value,
    const residency::Pool *const pool_value,
    residency::Authority *const authority_value) noexcept {
  if (phase != Phase::Ready || pipeline != nullptr || pool != nullptr ||
      authority != nullptr || pipeline_value == nullptr ||
      pool_value == nullptr || authority_value == nullptr || book == nullptr) {
    return false;
  }
  pipeline = pipeline_value;
  pool = pool_value;
  authority = authority_value;
  return true;
}

bool CpuGraphQuarantine::arm(
    const std::shared_ptr<CpuGraphQuarantine> &owner) noexcept {
  if (phase != Phase::Ready || pipeline == nullptr || pool == nullptr ||
      authority == nullptr || self != nullptr || book == nullptr ||
      owner == nullptr || owner.get() != this || owner->book != book) {
    return false;
  }
  self = owner;
  phase = Phase::Armed;
  return true;
}

void CpuGraphQuarantine::commit() noexcept {
  if (phase == Phase::Armed) {
    phase = Phase::Held;
    self.reset();
  }
}

void CpuGraphQuarantine::clear() noexcept {
  self.reset();
  phase = Phase::Ready;
}

void CpuGraphQuarantine::drop_handles() noexcept {
  if (book != nullptr) {
    book->drop_handles();
  }
}

} // namespace rund::compute::detail::graph_reduce
