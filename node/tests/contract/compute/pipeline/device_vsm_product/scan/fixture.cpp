#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "../../../../target/selection.hpp"

#include <node/runtime/compute/access.hpp>
#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <cstring>
#include <limits>
#include <span>
#include <utility>

namespace rund_node_test_device_vsm_product::scan_test {
namespace {

[[nodiscard]] std::uint64_t map_value(const std::uint64_t value,
                                      const ScanMapCase map) noexcept {
  if (map == ScanMapCase::Immediate) {
    return value + TypedAdd;
  }
  if (map == ScanMapCase::Canonical) {
    return (value ^ TypedMask) * TypedFactor + TypedAdd;
  }
  if (map == ScanMapCase::Branch) {
    return (value + 1u) + value * 2u;
  }
  return value;
}

[[nodiscard]] bool seed(PreparedScan &prepared, const bool overflow) noexcept {
  std::vector<std::uint64_t> input(
      static_cast<std::size_t>(prepared.element_count));
  prepared.expected.resize(input.size());
  std::uint64_t prefix = 0u;
  for (std::size_t index = 0u; index < input.size(); ++index) {
    input[index] = (index * 17u + 3u) % 97u;
  }
  if (overflow) {
    std::fill(input.begin(), input.end(), 0u);
    const std::size_t first = static_cast<std::size_t>(2u * PageElements);
    if (first + 1u >= input.size()) {
      return false;
    }
    input[first] = std::numeric_limits<std::uint64_t>::max();
    input[first + 1u] = 1u;
  }
  for (std::size_t index = 0u; index < input.size(); ++index) {
    const std::uint64_t mapped = map_value(input[index], prepared.map);
    if (prepared.operation == rund::kernel::ScanOp::InclusiveSum) {
      prefix += mapped;
    }
    prepared.expected[index] = prefix;
    if (prepared.operation == rund::kernel::ScanOp::ExclusiveSum) {
      prefix += mapped;
    }
  }
  return prepared.input != nullptr &&
         prepared.input->seed(std::as_bytes(std::span{input}));
}

} // namespace

bool PrepareScanProduct(const rund::compute::Backend backend,
                        const std::uint64_t pages,
                        const rund::kernel::ScanOp operation,
                        const bool overflow, const ScanMapCase map,
                        PreparedScan &prepared, bool &unavailable) {
  using namespace rund::compute;
  prepared = {};
  unavailable = false;
  const bool exclusive = operation == rund::kernel::ScanOp::ExclusiveSum;
  const std::uint64_t payload = PageElements - (exclusive ? 1u : 0u);
  const std::uint64_t elements = pages * payload - 3u;
  auto opened = open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    unavailable = opened.reason() == Reason::AdapterUnavailable;
    return false;
  }
  auto program = [&]() {
    if (map == ScanMapCase::None) {
      return on(*opened)
          .input<std::uint64_t>(PageElements)
          .branch([operation](auto values) {
            return values.scan(operation == rund::kernel::ScanOp::InclusiveSum
                                   ? Scan::InclusiveSum
                                   : Scan::ExclusiveSum);
          })
          .compile();
    }
    if (map == ScanMapCase::Immediate) {
      return on(*opened)
          .input<std::uint64_t>(PageElements)
          .branch([operation](auto values) {
            return values
                .map("device-vsm-map-scan-immediate",
                     capture([](auto value,
                                auto constant) { return value + constant; },
                             TypedAdd))
                .scan(operation == rund::kernel::ScanOp::InclusiveSum
                          ? Scan::InclusiveSum
                          : Scan::ExclusiveSum);
          })
          .compile();
    }
    if (map == ScanMapCase::Branch) {
      return on(*opened)
          .input<std::uint64_t>(PageElements)
          .branch([operation](auto values) {
            const auto left =
                values.map("device-vsm-branch-scan-left", [](auto value) {
                  return value + std::uint64_t{1u};
                });
            const auto right =
                values.map("device-vsm-branch-scan-right", [](auto value) {
                  return value * std::uint64_t{2u};
                });
            return zip(left, right)
                .map("device-vsm-branch-scan-join",
                     [](auto first, auto second) { return first + second; })
                .scan(operation == rund::kernel::ScanOp::InclusiveSum
                          ? Scan::InclusiveSum
                          : Scan::ExclusiveSum);
          })
          .compile();
    }
    return on(*opened)
        .input<std::uint64_t>(PageElements)
        .branch([operation](auto values) {
          return values
              .map("device-vsm-map-scan-canonical",
                   capture(
                       [](auto value, auto mask, auto factor, auto add) {
                         return (value ^ mask) * factor + add;
                       },
                       TypedMask, TypedFactor, TypedAdd))
              .scan(operation == rund::kernel::ScanOp::InclusiveSum
                        ? Scan::InclusiveSum
                        : Scan::ExclusiveSum);
        })
        .compile();
  }();
  prepared.input = std::make_shared<
      rund_node_test_persistent_product::PersistentProductBacking>(
      static_cast<std::size_t>(elements * sizeof(std::uint64_t)));
  prepared.output = std::make_shared<
      rund_node_test_persistent_product::PersistentProductBacking>(
      static_cast<std::size_t>(elements * sizeof(std::uint64_t)));
  prepared.element_count = elements;
  prepared.map = map;
  prepared.operation = operation;
  if (!seed(prepared, overflow)) {
    return false;
  }
  auto input = detail::make_virtual_buffer(
      elements, sizeof(std::uint64_t), detail::Type::U64, {}, prepared.input);
  auto output = detail::make_virtual_buffer(
      elements, sizeof(std::uint64_t), detail::Type::U64, {}, prepared.output);
  const auto program_state =
      program ? detail::ProgramAccess::state(*program) : nullptr;
  auto state =
      program_state != nullptr && input && output
          ? detail::prepare_virtual_pipeline(
                program_state, std::move(input).value(),
                std::move(output).value(), ResidencyConfig{})
          : Result<std::shared_ptr<detail::VirtualPipelineState>>::fail(
                Reason::PipelineInvalid);
  if (!state || state.value() == nullptr) {
    return false;
  }
  prepared.state = std::move(state).value();
  return prepared.state->geometry.route == detail::VirtualRoute::Scan &&
         prepared.state->pipeline != nullptr &&
         prepared.state->alternate_pipeline != nullptr &&
         prepared.state->pipeline->residency->stream().page_count() == pages &&
         prepared.state->pipeline->residency->stream().frame_capacity() == 2u;
}

bool SeedSafe(PreparedScan &prepared) noexcept { return seed(prepared, false); }

bool ExactScanOutput(const PreparedScan &prepared) noexcept {
  if (prepared.output == nullptr || prepared.expected.empty()) {
    return false;
  }
  std::vector<std::byte> bytes(prepared.expected.size() *
                               sizeof(std::uint64_t));
  return prepared.output->observe(bytes) &&
         std::memcmp(bytes.data(), prepared.expected.data(), bytes.size()) == 0;
}

} // namespace rund_node_test_device_vsm_product::scan_test

#endif
