#pragma once

#include "evidence.hpp"
#include "transfers.hpp"

#include <utility>

namespace rund::compute::detail::residency::execution {

// Strong lifetime handle for fixed Host/output/window state. Asynchronous
// owners retain a copy through their exact terminal callback.
class Sliding final {
public:
  Sliding() = default;

  [[nodiscard]] static Sliding
  create(SlidingInvocation, std::uint64_t authority_token,
         std::uint64_t run_generation, std::uint32_t host_input_capacity,
         std::uint32_t host_output_capacity) noexcept;
  [[nodiscard]] static Sliding
  create_bound(SlidingInvocation, std::uint64_t authority_token,
               std::uint64_t run_generation, std::uint32_t host_input_capacity,
               std::uint32_t host_output_capacity) noexcept;

  [[nodiscard]] explicit operator bool() const noexcept {
    return state_ != nullptr;
  }

  [[nodiscard]] bool project(std::uint64_t, std::span<residency::PageUse>,
                             SlidingProjection &) const noexcept;
  [[nodiscard]] bool reuse_fetch(const SlidingProjection &,
                                 std::span<residency::PageUse>,
                                 std::size_t) noexcept;
  [[nodiscard]] bool issue_fetch(const SlidingProjection &,
                                 std::span<residency::PageUse>, std::size_t,
                                 SlidingTicket &) noexcept;
  [[nodiscard]] bool fetch_terminal(const SlidingTicket &, Status, TerminalKind,
                                    bool may_write, std::uint64_t) noexcept;
  [[nodiscard]] bool issue_promote(const SlidingProjection &,
                                   std::span<residency::PageUse>,
                                   SlidingTicket &) noexcept;
  [[nodiscard]] bool promote_terminal(const SlidingTicket &, Status,
                                      TerminalKind, bool may_write,
                                      std::uint64_t) noexcept;
  [[nodiscard]] bool admit(const SlidingProjection &,
                           std::span<residency::PageUse>,
                           SlidingTicket &) noexcept;
  [[nodiscard]] bool native_terminal(const SlidingTicket &, Status,
                                     TerminalKind, bool may_write) noexcept;
  [[nodiscard]] bool issue_drain(const SlidingTicket &, std::size_t,
                                 SlidingTicket &) noexcept;
  [[nodiscard]] bool drain_terminal(const SlidingTicket &, Status, TerminalKind,
                                    bool may_write, std::uint64_t) noexcept;
  [[nodiscard]] bool issue_persist(const SlidingTicket &,
                                   SlidingTicket &) noexcept;
  [[nodiscard]] bool persist_terminal(const SlidingTicket &, Status,
                                      TerminalKind, bool may_write,
                                      std::uint64_t) noexcept;
  [[nodiscard]] bool invalidate(const SlidingTicket &) noexcept;
  [[nodiscard]] bool prepare_final(SlidingFinal &) noexcept;
  [[nodiscard]] bool accept_final(const SlidingFinal &,
                                  ExecutionSlidingReceipt &&,
                                  SlidingEvidence &) noexcept;
  [[nodiscard]] bool close_model(SlidingEvidence &) noexcept;
  [[nodiscard]] bool snapshot(SlidingEvidence &) const noexcept;
  [[nodiscard]] bool quiescent() const noexcept;
  [[nodiscard]] bool quarantined() const noexcept;

private:
  friend class ::rund::compute::detail::residency::Authority;
  friend class ::rund::compute::detail::residency::SlidingOwner;
  struct State;
  explicit Sliding(std::shared_ptr<State> state) noexcept
      : state_(std::move(state)) {}
  [[nodiscard]] static Sliding create_internal(SlidingInvocation, std::uint64_t,
                                               std::uint64_t, std::uint32_t,
                                               std::uint32_t,
                                               bool model_only) noexcept;
  [[nodiscard]] bool bind_authority(residency::Identity, std::uint64_t,
                                    std::uint64_t, std::uint64_t &) noexcept;

  std::shared_ptr<State> state_{};
};

} // namespace rund::compute::detail::residency::execution
