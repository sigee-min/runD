#pragma once

#include <rund/evidence/numeric/contract.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace rund::evidence {

class Numeric final {
public:
  enum class Code : std::uint8_t {
    Ok,
    NotBuilt,
    BadHeader,
    DuplicateField,
    MissingField,
    BadValue,
    HashInvalid,
  };

  constexpr Numeric() noexcept = default;

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok();
  }

  [[nodiscard]] constexpr bool ok() const noexcept { return code_ == Code::Ok; }

  [[nodiscard]] constexpr Code code() const noexcept { return code_; }
  [[nodiscard]] std::string_view error() const noexcept;
  [[nodiscard]] constexpr int exit_code() const noexcept {
    return ok() ? 0 : 1;
  }

  [[nodiscard]] constexpr std::optional<Contract> contract() const noexcept {
    return ok() ? std::optional<Contract>{contract_} : std::nullopt;
  }

  [[nodiscard]] constexpr Id id() const noexcept {
    return ok() ? identify(contract_) : Id{};
  }

  [[nodiscard]] constexpr bool strict_float() const noexcept {
    return ok() && contract_.arithmetic == Arithmetic::StrictFloatingPoint;
  }

  [[nodiscard]] std::uint64_t hash() const noexcept;

private:
  constexpr Numeric(const Contract contract, const Code code) noexcept
      : contract_(contract), code_(code) {}

  friend Numeric make(Contract contract) noexcept;
  friend Numeric decode(std::string_view encoded) noexcept;

  Contract contract_{};
  Code code_{Code::NotBuilt};
};

[[nodiscard]] Numeric make(Contract contract) noexcept;

[[nodiscard]] std::string encode(const Numeric &evidence);

[[nodiscard]] Numeric decode(std::string_view encoded) noexcept;

} // namespace rund::evidence
