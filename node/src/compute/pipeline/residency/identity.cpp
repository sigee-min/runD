#include "identity.hpp"

#include <cstdint>

namespace rund::compute::detail::residency {
namespace {

class Hash final {
public:
  Hash() noexcept {
    text("rund.compute.pipeline.residency");
    number(2u);
  }

  void byte(const std::uint8_t value) noexcept {
    lo_ ^= value;
    lo_ *= 1099511628211ull;
    hi_ ^= static_cast<std::uint8_t>(value + 0x9du);
    hi_ *= 14029467366897019727ull;
  }

  void number(const std::uint64_t value) noexcept {
    for (unsigned shift = 0u; shift != 64u; shift += 8u) {
      byte(static_cast<std::uint8_t>(value >> shift));
    }
  }

  void text(const char *value) noexcept {
    while (*value != '\0') {
      byte(static_cast<std::uint8_t>(*value++));
    }
    byte(0u);
  }

  [[nodiscard]] Identity finish() const noexcept {
    return Identity{.hi = hi_, .lo = lo_};
  }

private:
  std::uint64_t hi_{7809847782465536322ull};
  std::uint64_t lo_{1469598103934665603ull};
};

} // namespace

Identity IdentifyResidencyPlan(const std::uint64_t page_bytes,
                               const std::uint64_t page_count,
                               const std::uint64_t frame_capacity) noexcept {
  Hash hash{};
  hash.text("stream");
  hash.number(page_bytes);
  hash.number(page_count);
  hash.number(frame_capacity);
  return hash.finish();
}

Identity IdentifyResidencyPlan(const std::uint64_t page_bytes,
                               const std::uint32_t frame_capacity,
                               const std::vector<PageUse> &uses,
                               const std::vector<Epoch> &epochs) noexcept {
  Hash hash{};
  hash.text("graph");
  hash.number(page_bytes);
  hash.number(frame_capacity);
  hash.number(epochs.size());
  for (const Epoch &epoch : epochs) {
    hash.number(epoch.node);
    hash.number(epoch.tile);
    hash.number(epoch.use_count);
    for (std::size_t index = 0u; index < epoch.use_count; ++index) {
      const PageUse &use = uses[epoch.first_use + index];
      hash.number(use.key.resource);
      hash.number(use.key.page);
      hash.number(static_cast<std::uint8_t>(use.access));
    }
  }
  return hash.finish();
}

} // namespace rund::compute::detail::residency
