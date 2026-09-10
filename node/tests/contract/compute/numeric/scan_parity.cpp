#include "local.hpp"

#include <rund/compute.hpp>
#include <rund/compute/math.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

namespace {

template <class T> [[nodiscard]] T FromSmall(const std::int64_t value) {
  if constexpr (std::is_same_v<T, rund::compute::Fixed<1, 31>>) {
    return T::from_raw(static_cast<std::int32_t>(value));
  } else if constexpr (std::is_same_v<T, rund::compute::Fixed<1, 63>>) {
    return T::from_raw(value);
  } else {
    return static_cast<T>(value);
  }
}

template <class T> [[nodiscard]] std::uint64_t BitsOf(const T value) {
  if constexpr (std::is_same_v<T, rund::compute::Fixed<1, 31>>) {
    return static_cast<std::uint32_t>(value.raw());
  } else if constexpr (std::is_same_v<T, rund::compute::Fixed<1, 63>>) {
    return static_cast<std::uint64_t>(value.raw());
  } else if constexpr (sizeof(T) == sizeof(std::uint32_t)) {
    return static_cast<std::uint32_t>(value);
  } else {
    return static_cast<std::uint64_t>(value);
  }
}

struct ScanObservation final {
  std::vector<std::uint64_t> values;
  rund::compute::Stats stats;
};

template <class T>
[[nodiscard]] bool ObserveScan(const std::uint32_t workers,
                               ScanObservation &observation) {
  constexpr std::size_t kCount = 64u * 1024u + 17u;
  std::vector<T> input(kCount);
  for (std::size_t index = 0u; index < input.size(); ++index) {
    if constexpr (std::is_unsigned_v<T>) {
      input[index] = FromSmall<T>(static_cast<std::int64_t>(index % 5u));
    } else {
      input[index] = FromSmall<T>(static_cast<std::int64_t>(index % 5u) - 2);
    }
  }
  auto program = rund::compute::on(rund::compute::Target::cpu(workers))
                     .template map<T>(
                         "numeric-cross-tile-scan", input.size(),
                         [](auto value) {
                           if constexpr (rund::compute::detail::FixedValue<T>) {
                             return rund::compute::quantize<T>(value);
                           } else {
                             return value;
                           }
                         })
                     .scan(rund::compute::Scan::InclusiveSum)
                     .compile();
  if (!program) {
    return false;
  }
  auto job = program->resident(std::span<const T>{input});
  if (!job || !job->run()) {
    return false;
  }
  auto output = job->read();
  if (!output) {
    return false;
  }
  observation.values.reserve(output->size());
  for (const T value : *output) {
    observation.values.push_back(BitsOf(value));
  }
  observation.stats = job->stats();
  return true;
}

template <class T>
[[nodiscard]] bool CheckScanParity(const std::uint32_t workers) {
  ScanObservation one{};
  ScanObservation many{};
  if (!ObserveScan<T>(1u, one) || !ObserveScan<T>(workers, many)) {
    return false;
  }
  return one.values == many.values &&
         one.stats.graph_hash == many.stats.graph_hash &&
         one.stats.output_hash == many.stats.output_hash &&
         one.stats.worker_count == 1u && many.stats.worker_count == workers &&
         many.stats.tile_count > 1u &&
         (workers == 1u || many.stats.participating_workers > 1u);
}

} // namespace

bool CheckAllNumericScanParity(const std::uint32_t workers) {
  using rund::compute::Fixed;
  return CheckScanParity<std::int32_t>(workers) &&
         CheckScanParity<std::uint32_t>(workers) &&
         CheckScanParity<std::int64_t>(workers) &&
         CheckScanParity<std::uint64_t>(workers) &&
         CheckScanParity<Fixed<1, 31>>(workers) &&
         CheckScanParity<Fixed<1, 63>>(workers);
}
