#pragma once

#include <rund/replay/code.hpp>

#include <cstdint>
#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

namespace rund::node::replay {

struct KernelDesc {
  std::uint64_t run_key_hash = 0u;
  std::uint64_t program_hash = 0u;
  std::uint64_t phase_hash = 0u;
  std::uint64_t dispatch_hash = 0u;
  std::uint64_t reduction_hash = 0u;
  std::uint64_t capacity_hash = 0u;
  std::uint64_t output_hash = 0u;
  ::rund::replay::Code code = ::rund::replay::Code::KernelNotBuilt;
};

struct KernelRecord {
  ::rund::replay::Code code = ::rund::replay::Code::KernelNotBuilt;
  std::uint64_t run_key_hash = 0u;
  std::uint64_t program_hash = 0u;
  std::uint64_t phase_hash = 0u;
  std::uint64_t dispatch_hash = 0u;
  std::uint64_t reduction_hash = 0u;
  std::uint64_t capacity_hash = 0u;
  std::uint64_t output_hash = 0u;
  std::uint64_t record_hash = 0u;

  [[nodiscard]] bool ok() const noexcept {
    return code == ::rund::replay::Code::Ok;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
  [[nodiscard]] std::string_view error() const noexcept {
    return ::rund::replay::error(code);
  }
};

struct KernelCheck {
  ::rund::replay::Code code = ::rund::replay::Code::KernelNotChecked;
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

[[nodiscard]] KernelRecord MakeKernelRecord(KernelDesc desc);
[[nodiscard]] bool IsValidKernelRecord(const KernelRecord &record) noexcept;
[[nodiscard]] std::uint64_t
HashKernelRecord(const KernelRecord &record) noexcept;
[[nodiscard]] std::uint64_t
HashKernelRecords(std::span<const KernelRecord> records) noexcept;
[[nodiscard]] std::vector<std::byte>
EncodeKernelRecord(const KernelRecord &record);
[[nodiscard]] bool DecodeKernelRecord(std::span<const std::byte> encoded,
                                      KernelRecord &out);
[[nodiscard]] KernelCheck
CheckKernelRecord(const KernelRecord &expected,
                  const KernelRecord &actual) noexcept;

} // namespace rund::node::replay
