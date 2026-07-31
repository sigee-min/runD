#pragma once

#include <rund/replay/code.hpp>

#include <cstdint>
#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

namespace rund::node::replay {

struct AccelDesc {
  std::uint64_t graph_hash = 0u;
  std::uint64_t kernel_hash = 0u;
  std::uint64_t backend_hash = 0u;
  std::uint64_t caps_hash = 0u;
  std::uint64_t binding_hash = 0u;
  std::uint64_t buffer_shape_hash = 0u;
  std::uint64_t dispatch_hash = 0u;
  std::uint64_t output_hash = 0u;
  ::rund::replay::Code code = ::rund::replay::Code::AccelNotBuilt;
  std::uint64_t diagnostic_runtime_ns = 0u;
  std::uint64_t diagnostic_driver_hash = 0u;
  std::uint64_t diagnostic_cache_hash = 0u;
  std::uint64_t diagnostic_submit_count = 0u;
};

struct AccelRecord {
  ::rund::replay::Code code = ::rund::replay::Code::AccelNotBuilt;
  std::uint64_t graph_hash = 0u;
  std::uint64_t kernel_hash = 0u;
  std::uint64_t backend_hash = 0u;
  std::uint64_t caps_hash = 0u;
  std::uint64_t binding_hash = 0u;
  std::uint64_t buffer_shape_hash = 0u;
  std::uint64_t dispatch_hash = 0u;
  std::uint64_t output_hash = 0u;
  std::uint64_t diagnostic_runtime_ns = 0u;
  std::uint64_t diagnostic_driver_hash = 0u;
  std::uint64_t diagnostic_cache_hash = 0u;
  std::uint64_t diagnostic_submit_count = 0u;
  std::uint64_t semantic_hash = 0u;
  std::uint64_t diagnostic_hash = 0u;
  std::uint64_t record_hash = 0u;

  [[nodiscard]] bool ok() const noexcept {
    return code == ::rund::replay::Code::Ok;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
  [[nodiscard]] std::string_view error() const noexcept {
    return ::rund::replay::error(code);
  }
};

struct AccelCheck {
  ::rund::replay::Code code = ::rund::replay::Code::AccelNotChecked;
  std::uint64_t expected_hash = 0u;
  std::uint64_t actual_hash = 0u;

  [[nodiscard]] constexpr bool ok() const noexcept {
    return code == ::rund::replay::Code::Ok;
  }
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok();
  }
  [[nodiscard]] std::string_view error() const noexcept {
    return ::rund::replay::error(code);
  }
};

[[nodiscard]] AccelRecord MakeAccelRecord(AccelDesc desc);
[[nodiscard]] bool IsValidAccelRecord(const AccelRecord &record) noexcept;
[[nodiscard]] std::uint64_t HashAccelRecord(const AccelRecord &record) noexcept;
[[nodiscard]] std::uint64_t
HashAccelRecords(std::span<const AccelRecord> records) noexcept;
[[nodiscard]] std::vector<std::byte>
EncodeAccelRecord(const AccelRecord &record);
[[nodiscard]] bool DecodeAccelRecord(std::span<const std::byte> encoded,
                                     AccelRecord &out);
[[nodiscard]] AccelCheck CheckAccelRecord(const AccelRecord &expected,
                                          const AccelRecord &actual) noexcept;

} // namespace rund::node::replay
