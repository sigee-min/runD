#pragma once

#include <rund/reason.hpp>

#include <cstdint>
#include <memory>
#include <string_view>

namespace rund::detail::storage {
struct State;
}

namespace rund::storage {

struct Usage final {
  std::uint64_t physical_bytes = 0u;
  std::uint64_t allocated_bytes = 0u;
};

struct Report final {
  ReasonCode code = ReasonCode::StorageBudgetInvalid;
  std::uint64_t capacity_bytes = 0u;
  std::uint64_t physical_bytes = 0u;
  std::uint64_t allocated_bytes = 0u;
  std::uint64_t reserved_bytes = 0u;
  std::uint64_t available_bytes = 0u;
  std::uint64_t peak_physical_bytes = 0u;
  std::uint64_t peak_allocated_bytes = 0u;
  std::uint64_t peak_reserved_bytes = 0u;
  std::uint64_t peak_used_bytes = 0u;
  std::uint64_t reservation_count = 0u;
  std::uint64_t commit_count = 0u;
  std::uint64_t refund_count = 0u;
  std::uint64_t rejection_count = 0u;

  [[nodiscard]] constexpr bool ok() const noexcept {
    return code == ReasonCode::Ok;
  }
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok();
  }
  [[nodiscard]] std::string_view error() const noexcept {
    return ok() ? std::string_view{} : ReasonString(code);
  }
  [[nodiscard]] constexpr int exit_code() const noexcept {
    return ok() ? 0 : 1;
  }
};

class Status final {
public:
  constexpr Status() noexcept = default;

  [[nodiscard]] constexpr bool ok() const noexcept {
    return code_ == ReasonCode::Ok;
  }
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok();
  }
  [[nodiscard]] constexpr ReasonCode code() const noexcept { return code_; }
  [[nodiscard]] std::string_view error() const noexcept {
    return ok() ? std::string_view{} : ReasonString(code_);
  }
  [[nodiscard]] constexpr int exit_code() const noexcept {
    return ok() ? 0 : 1;
  }

private:
  explicit constexpr Status(const ReasonCode code) noexcept : code_(code) {}

  ReasonCode code_ = ReasonCode::StorageReservationInvalid;

  friend class Reservation;
};

class Reservation;

class Budget final {
public:
  Budget() noexcept = default;
  explicit Budget(std::uint64_t capacity_bytes) noexcept;

  Budget(const Budget &) noexcept = default;
  Budget &operator=(const Budget &) noexcept = default;
  Budget(Budget &&) noexcept = default;
  Budget &operator=(Budget &&) noexcept = default;

  [[nodiscard]] bool ok() const noexcept;
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
  [[nodiscard]] ReasonCode code() const noexcept;
  [[nodiscard]] std::string_view error() const noexcept {
    return ok() ? std::string_view{} : ReasonString(code());
  }
  [[nodiscard]] int exit_code() const noexcept { return ok() ? 0 : 1; }

  [[nodiscard]] Budget child(std::uint64_t capacity_bytes) const noexcept;
  [[nodiscard]] Reservation
  reserve(std::uint64_t max_allocated_bytes) const noexcept;
  [[nodiscard]] Report report() const noexcept;

private:
  Budget(std::shared_ptr<::rund::detail::storage::State> state,
         ReasonCode code) noexcept;

  std::shared_ptr<::rund::detail::storage::State> state_{};
  ReasonCode code_ = ReasonCode::StorageBudgetInvalid;
};

class Reservation final {
public:
  Reservation() noexcept = default;
  ~Reservation();

  Reservation(const Reservation &) = delete;
  Reservation &operator=(const Reservation &) = delete;
  Reservation(Reservation &&other) noexcept;
  Reservation &operator=(Reservation &&other) noexcept;

  [[nodiscard]] bool ok() const noexcept;
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
  [[nodiscard]] ReasonCode code() const noexcept;
  [[nodiscard]] std::string_view error() const noexcept {
    return ok() ? std::string_view{} : ReasonString(code());
  }
  [[nodiscard]] int exit_code() const noexcept { return ok() ? 0 : 1; }
  [[nodiscard]] bool committed() const noexcept;
  [[nodiscard]] std::uint64_t max_allocated_bytes() const noexcept;
  [[nodiscard]] Usage usage() const noexcept;

  // Splits one uncommitted reservation without performing another capacity
  // admission. The returned Reservation owns `max_allocated_bytes`; this
  // Reservation retains the exact remainder.
  [[nodiscard]] Reservation
  partition(std::uint64_t max_allocated_bytes) noexcept;
  [[nodiscard]] Status commit(Usage usage) noexcept;
  [[nodiscard]] Status refund() noexcept;

private:
  Reservation(std::shared_ptr<::rund::detail::storage::State> state,
              std::uint64_t max_allocated_bytes) noexcept;
  explicit Reservation(ReasonCode code) noexcept;

  void release() noexcept;

  std::shared_ptr<::rund::detail::storage::State> state_{};
  std::uint64_t max_allocated_bytes_ = 0u;
  Usage usage_{};
  ReasonCode code_ = ReasonCode::StorageReservationInvalid;
  bool committed_ = false;

  friend class Budget;
};

} // namespace rund::storage
