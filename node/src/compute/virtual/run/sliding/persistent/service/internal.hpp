#pragma once

#include "../../internal.hpp"

namespace rund::compute::detail::sliding_product_detail {

enum class PersistentServiceDisposition : std::uint8_t {
  Completed,
  KnownFailure,
  UnknownFailure,
};

struct PersistentServiceOutcome final {
  Status status{Status::fail(Reason::CompletionInvalid)};
  PersistentServiceDisposition disposition{
      PersistentServiceDisposition::UnknownFailure};
};

struct PersistentServiceCoordinate final {
  SlidingProductWork *work{};
  node::accel::detail::PersistentResidencySlidingServiceIdentity identity{};
  std::uint64_t coordinate{};
  std::uint64_t turn{};
  std::size_t slot{};
};

[[nodiscard]] PersistentServiceOutcome
service_persistent_coordinate(SlidingProductRun &, std::uint64_t) noexcept;
[[nodiscard]] PersistentServiceOutcome
make_persistent_unknown_failure(SlidingProductRun &, Status) noexcept;
[[nodiscard]] PersistentServiceOutcome
    make_persistent_known_failure(Status) noexcept;
[[nodiscard]] PersistentServiceOutcome make_persistent_completed() noexcept;
[[nodiscard]] bool initialize_persistent_service_coordinate(
    SlidingProductRun &, std::uint64_t, PersistentServiceCoordinate &) noexcept;
[[nodiscard]] bool
inject_persistent_unknown_failure(SlidingProductRun &,
                                  const PersistentServiceCoordinate &,
                                  PersistentServiceOutcome &) noexcept;
[[nodiscard]] PersistentServiceOutcome
admit_persistent_service_input(SlidingProductRun &,
                               PersistentServiceCoordinate &) noexcept;
[[nodiscard]] bool
select_persistent_service_native(SlidingProductRun &,
                                 PersistentServiceCoordinate &) noexcept;
[[nodiscard]] PersistentServiceOutcome
exchange_persistent_service_native(SlidingProductRun &,
                                   PersistentServiceCoordinate &) noexcept;
[[nodiscard]] PersistentServiceOutcome
service_persistent_coordinate_output(SlidingProductRun &,
                                     PersistentServiceCoordinate &) noexcept;
[[nodiscard]] Status service_persistent_known_suffix(SlidingProductRun &,
                                                     std::uint64_t,
                                                     Status) noexcept;
[[nodiscard]] rund::AccelCheck persistent_service_check(Status) noexcept;
[[nodiscard]] Status report_persistent_service_failure(
    SlidingProductRun &,
    const node::accel::detail::PersistentResidencySlidingServiceIdentity &,
    Status, node::accel::detail::NativeTerminal, bool) noexcept;
[[nodiscard]] Status acknowledge_persistent_service(
    SlidingProductRun &,
    const node::accel::detail::PersistentResidencySlidingServiceIdentity &,
    Status) noexcept;
[[nodiscard]] Status suppress_persistent_service_coordinate(SlidingProductRun &,
                                                            std::uint64_t,
                                                            Status) noexcept;
[[nodiscard]] Status terminal_persistent_native(
    SlidingProductRun &, SlidingProductWork &,
    const node::accel::detail::PersistentResidencySlidingDoneObservation
        &) noexcept;
void inject_persistent_service_unknown_once() noexcept;
[[nodiscard]] bool consume_persistent_service_unknown_injection() noexcept;

} // namespace rund::compute::detail::sliding_product_detail
