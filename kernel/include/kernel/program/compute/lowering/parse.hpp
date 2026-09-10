#pragma once

#include <kernel/program/compute/lowering/names.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace rund::kernel::compute_lowering_detail {

class Reader final {
public:
  explicit Reader(const std::vector<u8> &bytes) noexcept;

  [[nodiscard]] bool read_u8(u8 &value) noexcept;
  [[nodiscard]] bool read_u32(u32 &value) noexcept;
  [[nodiscard]] bool read_bytes(std::vector<u8> &out);
  [[nodiscard]] bool read_string(std::string &out);
  [[nodiscard]] bool done() const noexcept;
  [[nodiscard]] std::size_t remaining() const noexcept;

private:
  const std::vector<u8> *bytes_ = nullptr;
  std::size_t offset_ = 0u;
};

[[nodiscard]] ParsedIR ParseComputeIR(const ComputeIR &ir,
                                      const std::vector<u8> &canonical_bytes);
[[nodiscard]] ParsedIR ParseComputeIR(const ComputeIR &ir);

} // namespace rund::kernel::compute_lowering_detail
