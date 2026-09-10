#pragma once

#include "../registry.hpp"

namespace rund::compute::detail::residency {

using ExecutionFrame = Authority::Frame;

class ExecutionOwner final {
public:
  explicit ExecutionOwner(Authority &) noexcept;
  ExecutionOwner(const ExecutionOwner &) noexcept = default;
  ExecutionOwner &operator=(const ExecutionOwner &) = delete;

  [[nodiscard]] ExecutionLease
  begin_execution(const execution::Plan &plan) noexcept;
  [[nodiscard]] ExecutionLease
  begin_execution_window(const execution::Plan &plan) noexcept;
  [[nodiscard]] bool
  issue_execution(std::uint64_t token, std::uint64_t generation,
                  const execution::Plan &plan, const execution::Node &node,
                  std::uint64_t sequence, ExecutionTicket &ticket) noexcept;
  [[nodiscard]] bool terminal_execution(const ExecutionTicket &ticket,
                                        ExecutionTerminal terminal) noexcept;
  [[nodiscard]] bool release_execution(std::uint64_t token,
                                       std::uint64_t generation,
                                       const execution::Plan &plan,
                                       const execution::Release &release,
                                       std::uint64_t sequence) noexcept;
  [[nodiscard]] bool
  accept_execution_window(const execution::Plan &plan,
                          const execution::WindowEvidence &evidence) noexcept;
  [[nodiscard]] bool accept_execution_schedule(
      const execution::Plan &plan,
      const execution::ScheduleEvidence &evidence) noexcept;
  [[nodiscard]] bool
  accept_execution(const execution::Plan &plan,
                   const execution::NativeEvidence &evidence) noexcept;
  [[nodiscard]] bool reject_execution(std::uint64_t token,
                                      std::uint64_t generation,
                                      const execution::Plan &plan,
                                      Status failure) noexcept;
  [[nodiscard]] bool abandon_execution(std::uint64_t token,
                                       std::uint64_t generation,
                                       const execution::Plan &plan) noexcept;
  [[nodiscard]] bool
  abandon_execution_window(const execution::Plan &plan,
                           const execution::WindowEvidence &evidence) noexcept;
  [[nodiscard]] bool
  abandon_execution_stream(const execution::Plan &plan,
                           const execution::WindowEvidence &evidence) noexcept;
  [[nodiscard]] ExecutionClose
  abort_execution_stream(std::uint64_t token, std::uint64_t generation,
                         const execution::Plan &plan, Status failure) noexcept;
  [[nodiscard]] ExecutionClose
  close_execution(std::uint64_t token, std::uint64_t generation,
                  const execution::Plan &plan) noexcept;
  [[nodiscard]] ExecutionClose
  close_execution(const execution::Plan &plan,
                  const execution::Evidence &evidence) noexcept;

private:
  [[nodiscard]] bool validate_execution_issue_locked(
      std::uint64_t token, std::uint64_t generation, const execution::Plan &,
      const execution::Node &, std::uint64_t sequence, std::size_t &bank,
      std::size_t &phase) noexcept;
  [[nodiscard]] bool
  issue_execution_window_service_locked(const execution::Plan &,
                                        const execution::Node &,
                                        std::size_t bank) noexcept;
  [[nodiscard]] bool
  issue_execution_output_service_locked(const execution::Node &) noexcept;
  void finalize_execution_issue_locked(
      std::uint64_t token, std::uint64_t generation, const execution::Plan &,
      const execution::Node &, std::uint64_t sequence, std::size_t bank,
      std::size_t phase, bool window_service, ExecutionTicket &) noexcept;

  Authority &authority_;
};

} // namespace rund::compute::detail::residency
