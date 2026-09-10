#pragma once

#include "../../execution/evidence.hpp"
#include "../../execution/registration/state.hpp"
#include "../cache_model.hpp"

#include <rund/compute/status.hpp>

#include <cstdint>
#include <memory>
#include <utility>

namespace rund::compute::detail::residency {

class DirectRecurrenceOwner;

enum class DirectAbort : std::uint8_t {
  Invalid,
  Busy,
  Closed,
  Quarantined,
};

struct ResidentRecurrenceView final {
  std::uint64_t resource{};
  std::uint64_t bytes{};
  std::uint64_t offset_bytes{};
  std::uint64_t element_bytes{};
  std::uint64_t stride_bytes{};
  std::uint64_t count{};
  std::uint32_t usage{};

  [[nodiscard]] constexpr bool
  operator==(const ResidentRecurrenceView &) const noexcept = default;
};

struct ResidentRecurrenceBinding final {
  ResidentRecurrenceView view{};
  FrameRegion region{};
  std::uint64_t registration{};
};

struct DirectRecurrenceRequest final {
  std::shared_ptr<const void> proof_owner{};
  std::shared_ptr<const registration_detail::State> registration_state{};
  std::uint64_t proof_hi{};
  std::uint64_t proof_lo{};
  std::uint64_t iterations{};
};

class DirectRecurrenceLease final {
public:
  DirectRecurrenceLease() = default;
  DirectRecurrenceLease(const DirectRecurrenceLease &) = delete;
  DirectRecurrenceLease &operator=(const DirectRecurrenceLease &) = delete;
  DirectRecurrenceLease(DirectRecurrenceLease &&other) noexcept
      : proof_owner_(std::move(other.proof_owner_)), proof_hi_(other.proof_hi_),
        proof_lo_(other.proof_lo_), token_(other.token_),
        generation_(other.generation_), owner_(other.owner_),
        iterations_(other.iterations_),
        registration_state_(std::move(other.registration_state_)),
        registration_nonce_(other.registration_nonce_) {
    other.clear();
  }
  DirectRecurrenceLease &operator=(DirectRecurrenceLease &&) = delete;

  [[nodiscard]] explicit operator bool() const noexcept {
    return owner_ != 0u && proof_owner_ != nullptr &&
           registration_state_ != nullptr && registration_nonce_ != 0u &&
           registration_nonce_ == registration_state_->nonce() &&
           registration_state_->owner_matches(proof_owner_);
  }
  [[nodiscard]] std::uint64_t proof_hi() const noexcept { return proof_hi_; }
  [[nodiscard]] std::uint64_t proof_lo() const noexcept { return proof_lo_; }
  [[nodiscard]] std::uint64_t token() const noexcept { return token_; }
  [[nodiscard]] std::uint64_t generation() const noexcept {
    return generation_;
  }
  [[nodiscard]] std::uint64_t owner() const noexcept { return owner_; }
  [[nodiscard]] std::uint64_t iterations() const noexcept {
    return iterations_;
  }

private:
  friend class DirectRecurrenceOwner;
  void clear() noexcept {
    proof_owner_.reset();
    proof_hi_ = 0u;
    proof_lo_ = 0u;
    token_ = 0u;
    generation_ = 0u;
    owner_ = 0u;
    iterations_ = 0u;
    registration_state_.reset();
    registration_nonce_ = 0u;
  }

  std::shared_ptr<const void> proof_owner_{};
  std::uint64_t proof_hi_{};
  std::uint64_t proof_lo_{};
  std::uint64_t token_{};
  std::uint64_t generation_{};
  std::uint64_t owner_{};
  std::uint64_t iterations_{};
  std::shared_ptr<const registration_detail::State> registration_state_{};
  std::uint64_t registration_nonce_{};
};

class DirectRecurrenceFinal final {
public:
  DirectRecurrenceFinal() = default;
  DirectRecurrenceFinal(const DirectRecurrenceFinal &) = delete;
  DirectRecurrenceFinal &operator=(const DirectRecurrenceFinal &) = delete;
  DirectRecurrenceFinal(DirectRecurrenceFinal &&other) noexcept
      : proof_hi_(other.proof_hi_), proof_lo_(other.proof_lo_),
        token_(other.token_), generation_(other.generation_),
        owner_(other.owner_), iterations_(other.iterations_),
        completed_(other.completed_), status_(other.status_),
        terminal_(other.terminal_), may_write_(other.may_write_),
        registration_state_(std::move(other.registration_state_)),
        registration_nonce_(other.registration_nonce_) {
    other.clear();
  }
  DirectRecurrenceFinal &operator=(DirectRecurrenceFinal &&) = delete;
  [[nodiscard]] explicit operator bool() const noexcept {
    return owner_ != 0u && registration_state_ != nullptr &&
           registration_nonce_ != 0u &&
           registration_nonce_ == registration_state_->nonce();
  }

private:
  friend class DirectRecurrenceOwner;
  void clear() noexcept {
    proof_hi_ = 0u;
    proof_lo_ = 0u;
    token_ = 0u;
    generation_ = 0u;
    owner_ = 0u;
    iterations_ = 0u;
    completed_ = 0u;
    status_ = Status::fail(Reason::PipelineInvalid);
    terminal_ = execution::TerminalKind::Known;
    may_write_ = false;
    registration_state_.reset();
    registration_nonce_ = 0u;
  }

  std::uint64_t proof_hi_{};
  std::uint64_t proof_lo_{};
  std::uint64_t token_{};
  std::uint64_t generation_{};
  std::uint64_t owner_{};
  std::uint64_t iterations_{};
  std::uint64_t completed_{};
  Status status_{Status::fail(Reason::PipelineInvalid)};
  execution::TerminalKind terminal_{execution::TerminalKind::Known};
  bool may_write_{};
  std::shared_ptr<const registration_detail::State> registration_state_{};
  std::uint64_t registration_nonce_{};
};

using DirectRecurrencePublication = void (*)(void *, bool) noexcept;

} // namespace rund::compute::detail::residency
