#include <rund/storage.hpp>

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <limits>
#include <mutex>
#include <new>
#include <utility>

namespace rund::detail::storage {

struct Hierarchy final {
  std::mutex gate{};
};

struct State final {
  State(std::shared_ptr<Hierarchy> hierarchy_value,
        std::shared_ptr<State> parent_value,
        const std::uint64_t capacity_value) noexcept
      : hierarchy(std::move(hierarchy_value)), parent(std::move(parent_value)),
        capacity_bytes(capacity_value) {}

  std::shared_ptr<Hierarchy> hierarchy{};
  std::shared_ptr<State> parent{};
  std::uint64_t capacity_bytes = 0u;
  std::uint64_t physical_bytes = 0u;
  std::uint64_t allocated_bytes = 0u;
  std::uint64_t reserved_bytes = 0u;
  std::uint64_t peak_physical_bytes = 0u;
  std::uint64_t peak_allocated_bytes = 0u;
  std::uint64_t peak_reserved_bytes = 0u;
  std::uint64_t peak_used_bytes = 0u;
  std::uint64_t reservation_count = 0u;
  std::uint64_t commit_count = 0u;
  std::uint64_t refund_count = 0u;
  std::uint64_t rejection_count = 0u;
};

} // namespace rund::detail::storage

namespace rund::storage {
namespace {

using State = ::rund::detail::storage::State;

void Increment(std::uint64_t &value) noexcept {
  if (value != std::numeric_limits<std::uint64_t>::max()) {
    ++value;
  }
}

void Reject(State *state) noexcept {
  for (; state != nullptr; state = state->parent.get()) {
    Increment(state->rejection_count);
  }
}

[[nodiscard]] bool Admits(const State &state,
                          const std::uint64_t bytes) noexcept {
  if (state.allocated_bytes > state.capacity_bytes ||
      state.reserved_bytes > state.capacity_bytes - state.allocated_bytes) {
    return false;
  }
  return bytes <=
         state.capacity_bytes - state.allocated_bytes - state.reserved_bytes;
}

[[nodiscard]] bool Admits(const State &state, const Usage usage) noexcept {
  return rund::kernel::checked::add(state.physical_bytes,
                                    usage.physical_bytes) &&
         rund::kernel::checked::add(state.allocated_bytes,
                                    usage.allocated_bytes);
}

void Reserve(State *state, const std::uint64_t bytes) noexcept {
  for (; state != nullptr; state = state->parent.get()) {
    state->reserved_bytes += bytes;
    state->peak_reserved_bytes =
        std::max(state->peak_reserved_bytes, state->reserved_bytes);
    state->peak_used_bytes = std::max(
        state->peak_used_bytes, state->reserved_bytes + state->allocated_bytes);
    Increment(state->reservation_count);
  }
}

void Commit(State *state, const std::uint64_t reserved,
            const Usage usage) noexcept {
  for (; state != nullptr; state = state->parent.get()) {
    state->reserved_bytes -= reserved;
    state->allocated_bytes += usage.allocated_bytes;
    state->physical_bytes += usage.physical_bytes;
    state->peak_allocated_bytes =
        std::max(state->peak_allocated_bytes, state->allocated_bytes);
    state->peak_physical_bytes =
        std::max(state->peak_physical_bytes, state->physical_bytes);
    state->peak_used_bytes = std::max(
        state->peak_used_bytes, state->reserved_bytes + state->allocated_bytes);
    Increment(state->commit_count);
  }
}

void Refund(State *state, const std::uint64_t reserved, const Usage usage,
            const bool committed) noexcept {
  for (; state != nullptr; state = state->parent.get()) {
    if (committed) {
      state->allocated_bytes -= usage.allocated_bytes;
      state->physical_bytes -= usage.physical_bytes;
    } else {
      state->reserved_bytes -= reserved;
    }
    Increment(state->refund_count);
  }
}

} // namespace

Budget::Budget(const std::uint64_t capacity_bytes) noexcept {
  if (capacity_bytes == 0u) {
    return;
  }
  try {
    auto hierarchy = std::make_shared<::rund::detail::storage::Hierarchy>();
    state_ =
        std::make_shared<State>(std::move(hierarchy), nullptr, capacity_bytes);
    code_ = ReasonCode::Ok;
  } catch (const std::bad_alloc &) {
    state_.reset();
    code_ = ReasonCode::StorageBudgetAllocationFailed;
  }
}

Budget::Budget(std::shared_ptr<State> state, const ReasonCode code) noexcept
    : state_(std::move(state)), code_(code) {}

bool Budget::ok() const noexcept {
  return state_ != nullptr && code_ == ReasonCode::Ok;
}

ReasonCode Budget::code() const noexcept {
  return state_ == nullptr && code_ == ReasonCode::Ok
             ? ReasonCode::StorageBudgetInvalid
             : code_;
}

Budget Budget::child(const std::uint64_t capacity_bytes) const noexcept {
  if (!ok()) {
    return Budget{{}, code()};
  }
  if (capacity_bytes == 0u || capacity_bytes > state_->capacity_bytes) {
    return Budget{{}, ReasonCode::StorageBudgetInvalid};
  }
  try {
    return Budget{
        std::make_shared<State>(state_->hierarchy, state_, capacity_bytes),
        ReasonCode::Ok};
  } catch (const std::bad_alloc &) {
    return Budget{{}, ReasonCode::StorageBudgetAllocationFailed};
  }
}

Reservation
Budget::reserve(const std::uint64_t max_allocated_bytes) const noexcept {
  if (!ok()) {
    return Reservation{code()};
  }
  std::lock_guard lock{state_->hierarchy->gate};
  for (State *current = state_.get(); current != nullptr;
       current = current->parent.get()) {
    if (!Admits(*current, max_allocated_bytes)) {
      Reject(state_.get());
      return Reservation{ReasonCode::StorageCapacityExceeded};
    }
  }
  Reserve(state_.get(), max_allocated_bytes);
  return Reservation{state_, max_allocated_bytes};
}

Report Budget::report() const noexcept {
  if (!ok()) {
    return Report{.code = code()};
  }
  std::lock_guard lock{state_->hierarchy->gate};
  const std::uint64_t used = state_->allocated_bytes + state_->reserved_bytes;
  return Report{
      .code = ReasonCode::Ok,
      .capacity_bytes = state_->capacity_bytes,
      .physical_bytes = state_->physical_bytes,
      .allocated_bytes = state_->allocated_bytes,
      .reserved_bytes = state_->reserved_bytes,
      .available_bytes = state_->capacity_bytes - used,
      .peak_physical_bytes = state_->peak_physical_bytes,
      .peak_allocated_bytes = state_->peak_allocated_bytes,
      .peak_reserved_bytes = state_->peak_reserved_bytes,
      .peak_used_bytes = state_->peak_used_bytes,
      .reservation_count = state_->reservation_count,
      .commit_count = state_->commit_count,
      .refund_count = state_->refund_count,
      .rejection_count = state_->rejection_count,
  };
}

Reservation::Reservation(std::shared_ptr<State> state,
                         const std::uint64_t max_allocated_bytes) noexcept
    : state_(std::move(state)), max_allocated_bytes_(max_allocated_bytes),
      code_(ReasonCode::Ok) {}

Reservation::Reservation(const ReasonCode code) noexcept : code_(code) {}

Reservation::~Reservation() { release(); }

Reservation::Reservation(Reservation &&other) noexcept
    : state_(std::move(other.state_)),
      max_allocated_bytes_(other.max_allocated_bytes_), usage_(other.usage_),
      code_(other.code_), committed_(other.committed_) {
  other.max_allocated_bytes_ = 0u;
  other.usage_ = {};
  other.code_ = ReasonCode::StorageReservationInvalid;
  other.committed_ = false;
}

Reservation &Reservation::operator=(Reservation &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  release();
  state_ = std::move(other.state_);
  max_allocated_bytes_ = other.max_allocated_bytes_;
  usage_ = other.usage_;
  code_ = other.code_;
  committed_ = other.committed_;
  other.max_allocated_bytes_ = 0u;
  other.usage_ = {};
  other.code_ = ReasonCode::StorageReservationInvalid;
  other.committed_ = false;
  return *this;
}

bool Reservation::ok() const noexcept {
  return state_ != nullptr && code_ == ReasonCode::Ok;
}

ReasonCode Reservation::code() const noexcept {
  return state_ == nullptr && code_ == ReasonCode::Ok
             ? ReasonCode::StorageReservationInvalid
             : code_;
}

bool Reservation::committed() const noexcept { return ok() && committed_; }

std::uint64_t Reservation::max_allocated_bytes() const noexcept {
  return ok() ? max_allocated_bytes_ : 0u;
}

Usage Reservation::usage() const noexcept {
  return committed() ? usage_ : Usage{};
}

Reservation
Reservation::partition(const std::uint64_t max_allocated_bytes) noexcept {
  if (!ok() || committed_) {
    return Reservation{ReasonCode::StorageReservationInvalid};
  }
  std::lock_guard lock{state_->hierarchy->gate};
  if (max_allocated_bytes > max_allocated_bytes_) {
    Reject(state_.get());
    return Reservation{ReasonCode::StorageReservationInvalid};
  }
  max_allocated_bytes_ -= max_allocated_bytes;
  return Reservation{state_, max_allocated_bytes};
}

Status Reservation::commit(const Usage usage) noexcept {
  if (!ok() || committed_) {
    return Status{ReasonCode::StorageReservationInvalid};
  }
  std::lock_guard lock{state_->hierarchy->gate};
  if (usage.allocated_bytes > max_allocated_bytes_) {
    Reject(state_.get());
    return Status{ReasonCode::StorageCommitInvalid};
  }
  for (State *current = state_.get(); current != nullptr;
       current = current->parent.get()) {
    if (!Admits(*current, usage)) {
      Reject(state_.get());
      return Status{ReasonCode::StorageCommitInvalid};
    }
  }
  Commit(state_.get(), max_allocated_bytes_, usage);
  usage_ = usage;
  committed_ = true;
  return Status{ReasonCode::Ok};
}

Status Reservation::refund() noexcept {
  if (!ok()) {
    return Status{ReasonCode::StorageReservationInvalid};
  }
  release();
  return Status{ReasonCode::Ok};
}

void Reservation::release() noexcept {
  if (state_ == nullptr) {
    return;
  }
  const std::shared_ptr<State> state = state_;
  {
    std::lock_guard lock{state->hierarchy->gate};
    Refund(state.get(), max_allocated_bytes_, usage_, committed_);
  }
  state_.reset();
  max_allocated_bytes_ = 0u;
  usage_ = {};
  code_ = ReasonCode::StorageReservationInvalid;
  committed_ = false;
}

} // namespace rund::storage
