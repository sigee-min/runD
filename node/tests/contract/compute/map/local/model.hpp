#pragma once

#include <cstdint>
#include <memory>

namespace rund::compute {
enum class Backend : std::uint8_t;
namespace detail {
struct DeviceState;
} // namespace detail
namespace graph {
struct Fingerprint;
} // namespace graph
} // namespace rund::compute

namespace compute_map_contract {

[[nodiscard]] bool Canonical();
[[nodiscard]] bool Replay();
[[nodiscard]] bool
Liveness(const std::shared_ptr<rund::compute::detail::DeviceState> &device);
[[nodiscard]] bool
Envelope(const std::shared_ptr<rund::compute::detail::DeviceState> &device,
         rund::compute::Backend backend,
         rund::compute::graph::Fingerprint &reference32,
         rund::compute::graph::Fingerprint &reference64);
[[nodiscard]] bool Carrier(rund::compute::Backend backend,
                           rund::compute::graph::Fingerprint &envelope32,
                           rund::compute::graph::Fingerprint &envelope64);
int Run();

} // namespace compute_map_contract
