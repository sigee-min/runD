#include "identity.hpp"

namespace rund::compute::detail {

namespace {

class MaterializationHash final {
public:
  MaterializationHash() noexcept { text("rund.compute.vsm.frame.v1"); }

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

  [[nodiscard]] residency::Identity finish() const noexcept {
    return residency::Identity{.hi = hi_, .lo = lo_};
  }

private:
  std::uint64_t hi_{7809847782465536322ull};
  std::uint64_t lo_{1469598103934665603ull};
};

} // namespace

[[nodiscard]] residency::Identity input_materialization_identity(
    const Type type, const FixedFormat format, const std::uint64_t page_bytes,
    const std::uint64_t payload_bytes, const std::uint64_t prefix_bytes,
    const std::uint64_t frame_elements, const std::uint32_t boundary) noexcept {
  MaterializationHash hash{};
  hash.text("backing-input");
  hash.number(static_cast<std::uint8_t>(type));
  hash.number(format.integer_bits);
  hash.number(format.fraction_bits);
  hash.number(static_cast<std::uint8_t>(format.rounding));
  hash.number(static_cast<std::uint8_t>(format.overflow));
  hash.number(static_cast<std::uint8_t>(format.approximation));
  hash.number(page_bytes);
  hash.number(payload_bytes);
  hash.number(prefix_bytes);
  hash.number(frame_elements);
  hash.number(boundary);
  return hash.finish();
}

[[nodiscard]] residency::Identity
output_materialization_identity(const Type type, const FixedFormat format,
                                const std::uint64_t page_bytes,
                                const std::uint64_t frame_elements) noexcept {
  MaterializationHash hash{};
  hash.text("backing-output");
  hash.number(static_cast<std::uint8_t>(type));
  hash.number(format.integer_bits);
  hash.number(format.fraction_bits);
  hash.number(static_cast<std::uint8_t>(format.rounding));
  hash.number(static_cast<std::uint8_t>(format.overflow));
  hash.number(static_cast<std::uint8_t>(format.approximation));
  hash.number(page_bytes);
  hash.number(frame_elements);
  return hash.finish();
}

[[nodiscard]] residency::Identity
transient_materialization_identity(const residency::Identity graph,
                                   const std::uint32_t resource,
                                   const std::uint64_t domain) noexcept {
  MaterializationHash hash{};
  hash.text("graph-transient");
  hash.number(graph.hi);
  hash.number(graph.lo);
  hash.number(resource);
  hash.number(domain);
  return hash.finish();
}

} // namespace rund::compute::detail
