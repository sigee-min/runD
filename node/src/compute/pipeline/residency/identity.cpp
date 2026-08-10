#include "identity.hpp"

#include <cstdint>

namespace rund::compute::detail::residency {
namespace {

class Hash final {
public:
  Hash() noexcept {
    text("rund.compute.pipeline.residency");
    number(1u);
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
                               const std::uint64_t slot_capacity) noexcept {
  Hash hash{};
  hash.number(page_bytes);
  hash.number(page_count);
  hash.number(slot_capacity);
  return hash.finish();
}

} // namespace rund::compute::detail::residency
