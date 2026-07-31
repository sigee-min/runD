#pragma once

#include <rund/compute/fixed.hpp>
#include <rund/compute/graph/info.hpp>

#include <cstdint>

namespace rund::compute::detail {

class PipelineHash final {
public:
  PipelineHash() noexcept;

  void byte(std::uint8_t value) noexcept;
  void number(std::uint64_t value) noexcept;
  void text(const char *value) noexcept;
  void format(FixedFormat value) noexcept;
  [[nodiscard]] graph::Fingerprint finish() const noexcept;

private:
  std::uint64_t hi_{7809847782465536322ull};
  std::uint64_t lo_{1469598103934665603ull};
};

[[nodiscard]] bool valid_format(Type type, FixedFormat format) noexcept;
[[nodiscard]] bool typed_format_matches(Type type, FixedFormat typed,
                                        FixedFormat slot) noexcept;

} // namespace rund::compute::detail
