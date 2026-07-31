#include <rund/evidence/numeric.hpp>

#include "../host/hash/fields.hpp"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <system_error>

namespace rund::evidence {
namespace {

constexpr std::string_view kHeader = "rund.numeric.evidence\n";
constexpr std::uint16_t kAllFields = (1u << 11u) - 1u;

enum class DecodeFailure : std::uint8_t {
  None,
  BadHeader,
  DuplicateField,
  MissingField,
  BadValue,
  HashInvalid,
};

struct DecodedFields final {
  Contract contract{};
  std::uint64_t hash = 0u;
  std::uint16_t seen = 0u;
  DecodeFailure failure = DecodeFailure::None;
};

void append_line(std::string &out, const std::string_view key,
                 const std::uint64_t value) {
  out.append(key);
  out.push_back('=');
  out.append(std::to_string(value));
  out.push_back('\n');
}

[[nodiscard]] bool parse_unsigned(const std::string_view text,
                                  std::uint64_t &out) noexcept {
  const char *const first = text.data();
  const char *const last = first + text.size();
  const std::from_chars_result parsed = std::from_chars(first, last, out);
  return parsed.ec == std::errc{} && parsed.ptr == last;
}

[[nodiscard]] bool claim_field(DecodedFields &fields,
                               const std::uint16_t bit) noexcept {
  if ((fields.seen & bit) != 0u) {
    fields.failure = DecodeFailure::DuplicateField;
    return false;
  }
  fields.seen = static_cast<std::uint16_t>(fields.seen | bit);
  return true;
}

[[nodiscard]] bool parse_byte(const std::string_view text,
                              std::uint8_t &out) noexcept {
  std::uint64_t value = 0u;
  if (!parse_unsigned(text, value) ||
      value > std::numeric_limits<std::uint8_t>::max()) {
    return false;
  }
  out = static_cast<std::uint8_t>(value);
  return true;
}

[[nodiscard]] std::uint64_t hash_contract(const Contract contract) noexcept {
  constexpr std::uint64_t kCanonicalBytes =
      sizeof(std::uint32_t) + (9u * sizeof(std::uint8_t));
  ::rund::node::host_detail::CanonicalByteHasher hash{kCanonicalBytes};
  hash.AppendU32Le(detail::contract::schema_tag);
  hash.AppendByte(static_cast<std::byte>(contract.domain));
  hash.AppendByte(static_cast<std::byte>(contract.arithmetic));
  hash.AppendByte(static_cast<std::byte>(contract.authority));
  hash.AppendByte(static_cast<std::byte>(contract.determinism));
  hash.AppendByte(static_cast<std::byte>(contract.integer_bits));
  hash.AppendByte(static_cast<std::byte>(contract.fraction_bits));
  hash.AppendByte(static_cast<std::byte>(contract.rounding));
  hash.AppendByte(static_cast<std::byte>(contract.overflow));
  hash.AppendByte(static_cast<std::byte>(contract.approximation));
  return hash.Finish().value;
}

template <typename Assign>
[[nodiscard]] bool
parse_byte_field(DecodedFields &fields, const std::uint16_t bit,
                 const std::string_view text, Assign &&assign) noexcept {
  if (!claim_field(fields, bit)) {
    return false;
  }
  std::uint8_t value = 0u;
  if (!parse_byte(text, value)) {
    fields.failure = DecodeFailure::BadValue;
    return false;
  }
  assign(value);
  return true;
}

[[nodiscard]] bool parse_field(DecodedFields &fields,
                               const std::string_view key,
                               const std::string_view text) noexcept {
  if (key == "schema") {
    constexpr std::uint16_t bit = 1u << 0u;
    std::uint64_t value = 0u;
    if (!claim_field(fields, bit)) {
      return false;
    }
    if (!parse_unsigned(text, value) ||
        value != detail::contract::schema_tag) {
      fields.failure = DecodeFailure::BadValue;
      return false;
    }
    return true;
  }
  if (key == "domain") {
    return parse_byte_field(fields, 1u << 1u, text, [&](const auto value) {
      fields.contract.domain = static_cast<Domain>(value);
    });
  }
  if (key == "arithmetic") {
    return parse_byte_field(fields, 1u << 2u, text, [&](const auto value) {
      fields.contract.arithmetic = static_cast<Arithmetic>(value);
    });
  }
  if (key == "authority") {
    return parse_byte_field(fields, 1u << 3u, text, [&](const auto value) {
      fields.contract.authority = static_cast<Authority>(value);
    });
  }
  if (key == "determinism") {
    return parse_byte_field(fields, 1u << 4u, text, [&](const auto value) {
      fields.contract.determinism = static_cast<Determinism>(value);
    });
  }
  if (key == "integer_bits") {
    return parse_byte_field(fields, 1u << 5u, text, [&](const auto value) {
      fields.contract.integer_bits = value;
    });
  }
  if (key == "fraction_bits") {
    return parse_byte_field(fields, 1u << 6u, text, [&](const auto value) {
      fields.contract.fraction_bits = value;
    });
  }
  if (key == "rounding") {
    return parse_byte_field(fields, 1u << 7u, text, [&](const auto value) {
      fields.contract.rounding = static_cast<Rounding>(value);
    });
  }
  if (key == "overflow") {
    return parse_byte_field(fields, 1u << 8u, text, [&](const auto value) {
      fields.contract.overflow = static_cast<Overflow>(value);
    });
  }
  if (key == "approximation") {
    return parse_byte_field(fields, 1u << 9u, text, [&](const auto value) {
      fields.contract.approximation = static_cast<Approximation>(value);
    });
  }
  if (key == "hash") {
    constexpr std::uint16_t bit = 1u << 10u;
    if (!claim_field(fields, bit)) {
      return false;
    }
    if (!parse_unsigned(text, fields.hash)) {
      fields.failure = DecodeFailure::BadValue;
      return false;
    }
    return true;
  }
  fields.failure = DecodeFailure::BadHeader;
  return false;
}

[[nodiscard]] DecodedFields
decode_fields(const std::string_view encoded) noexcept {
  DecodedFields fields{};
  if (!encoded.starts_with(kHeader)) {
    fields.failure = DecodeFailure::BadHeader;
    return fields;
  }

  std::size_t cursor = kHeader.size();
  while (cursor < encoded.size()) {
    std::size_t line_end = encoded.find('\n', cursor);
    if (line_end == std::string_view::npos) {
      line_end = encoded.size();
    }
    const std::string_view line = encoded.substr(cursor, line_end - cursor);
    const std::size_t equals = line.find('=');
    if (line.empty() || equals == std::string_view::npos || equals == 0u ||
        equals + 1u == line.size() ||
        !parse_field(fields, line.substr(0u, equals),
                     line.substr(equals + 1u))) {
      if (fields.failure == DecodeFailure::None) {
        fields.failure = DecodeFailure::BadHeader;
      }
      return fields;
    }
    cursor = line_end + 1u;
  }

  if (fields.seen != kAllFields) {
    fields.failure = DecodeFailure::MissingField;
    return fields;
  }
  if (!valid(fields.contract)) {
    fields.failure = DecodeFailure::BadValue;
    return fields;
  }
  if (fields.hash != hash_contract(fields.contract)) {
    fields.failure = DecodeFailure::HashInvalid;
    return fields;
  }
  return fields;
}

} // namespace

std::string_view Numeric::error() const noexcept {
  switch (code_) {
  case Code::Ok:
    return {};
  case Code::NotBuilt:
    return "numeric_evidence_not_built";
  case Code::BadHeader:
    return "numeric_evidence_bad_header";
  case Code::DuplicateField:
    return "numeric_evidence_duplicate_field";
  case Code::MissingField:
    return "numeric_evidence_missing_field";
  case Code::BadValue:
    return "numeric_evidence_bad_value";
  case Code::HashInvalid:
    return "numeric_evidence_hash_invalid";
  }
  return "numeric_evidence_bad_value";
}

std::uint64_t Numeric::hash() const noexcept {
  return ok() ? hash_contract(contract_) : 0u;
}

Numeric make(const Contract contract) noexcept {
  return valid(contract) ? Numeric{contract, Numeric::Code::Ok}
                         : Numeric{contract, Numeric::Code::BadValue};
}

std::string encode(const Numeric &evidence) {
  const std::optional<Contract> contract = evidence.contract();
  if (!contract) {
    return {};
  }

  std::string encoded{kHeader};
  encoded.reserve(256u);
  append_line(encoded, "schema", detail::contract::schema_tag);
  append_line(encoded, "domain", static_cast<std::uint8_t>(contract->domain));
  append_line(encoded, "arithmetic",
              static_cast<std::uint8_t>(contract->arithmetic));
  append_line(encoded, "authority",
              static_cast<std::uint8_t>(contract->authority));
  append_line(encoded, "determinism",
              static_cast<std::uint8_t>(contract->determinism));
  append_line(encoded, "integer_bits", contract->integer_bits);
  append_line(encoded, "fraction_bits", contract->fraction_bits);
  append_line(encoded, "rounding",
              static_cast<std::uint8_t>(contract->rounding));
  append_line(encoded, "overflow",
              static_cast<std::uint8_t>(contract->overflow));
  append_line(encoded, "approximation",
              static_cast<std::uint8_t>(contract->approximation));
  append_line(encoded, "hash", evidence.hash());
  return encoded;
}

Numeric decode(const std::string_view encoded) noexcept {
  const DecodedFields fields = decode_fields(encoded);
  switch (fields.failure) {
  case DecodeFailure::None:
    return Numeric{fields.contract, Numeric::Code::Ok};
  case DecodeFailure::BadHeader:
    return Numeric{{}, Numeric::Code::BadHeader};
  case DecodeFailure::DuplicateField:
    return Numeric{{}, Numeric::Code::DuplicateField};
  case DecodeFailure::MissingField:
    return Numeric{{}, Numeric::Code::MissingField};
  case DecodeFailure::BadValue:
    return Numeric{{}, Numeric::Code::BadValue};
  case DecodeFailure::HashInvalid:
    return Numeric{{}, Numeric::Code::HashInvalid};
  }
  return Numeric{{}, Numeric::Code::BadValue};
}

} // namespace rund::evidence
