#include "test/assert.hpp"

#include <rund/evidence.hpp>

#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>

namespace {

constexpr std::uint64_t kStrictF64Id = 11944853433465158127ull;
constexpr std::uint64_t kStrictF64Hash = 14337942912644298094ull;
constexpr std::string_view kStrictF64Text = "rund.numeric.evidence\n"
                                            "schema=1\n"
                                            "domain=6\n"
                                            "arithmetic=4\n"
                                            "authority=0\n"
                                            "determinism=0\n"
                                            "integer_bits=0\n"
                                            "fraction_bits=0\n"
                                            "rounding=3\n"
                                            "overflow=0\n"
                                            "approximation=0\n"
                                            "hash=14337942912644298094\n";

static_assert(noexcept(rund::evidence::make(rund::evidence::Contract{})));
static_assert(noexcept(rund::evidence::decode({})));
static_assert(std::is_trivially_copyable_v<rund::evidence::Numeric>);
static_assert(std::is_trivially_destructible_v<rund::evidence::Numeric>);
static_assert(sizeof(rund::evidence::Contract) == 9u);
static_assert(sizeof(rund::evidence::Numeric) == 10u);
static_assert(alignof(rund::evidence::Numeric) == 1u);

[[nodiscard]] bool insert_duplicate_line(std::string &text,
                                         const std::string_view key) {
  const std::size_t begin = text.find(key);
  if (begin == std::string::npos) {
    return false;
  }
  const std::size_t end = text.find('\n', begin);
  if (end == std::string::npos) {
    return false;
  }
  text.insert(end + 1u, text.substr(begin, end - begin + 1u));
  return true;
}

[[nodiscard]] bool remove_line(std::string &text, const std::string_view key) {
  const std::size_t begin = text.find(key);
  if (begin == std::string::npos) {
    return false;
  }
  const std::size_t end = text.find('\n', begin);
  if (end == std::string::npos) {
    return false;
  }
  text.erase(begin, end - begin + 1u);
  return true;
}

[[nodiscard]] bool replace_line(std::string &text, const std::string_view key,
                                const std::string_view replacement) {
  const std::size_t begin = text.find(key);
  if (begin == std::string::npos) {
    return false;
  }
  const std::size_t end = text.find('\n', begin);
  if (end == std::string::npos) {
    return false;
  }
  text.replace(begin, end - begin, replacement);
  return true;
}

void assert_rejected(const std::string_view text,
                     const rund::evidence::Numeric::Code code,
                     const std::string_view error) {
  const rund::evidence::Numeric decoded = rund::evidence::decode(text);
  TEST_ASSERT(!decoded);
  TEST_ASSERT(!decoded.ok());
  TEST_ASSERT(decoded.code() == code);
  TEST_ASSERT(decoded.error() == error);
  TEST_ASSERT(decoded.exit_code() == 1);
  TEST_ASSERT(!decoded.contract().has_value());
  TEST_ASSERT(decoded.id().value == 0u);
  TEST_ASSERT(!decoded.strict_float());
  TEST_ASSERT(decoded.hash() == 0u);
}

void assert_round_trip(const rund::evidence::Contract contract) {
  const rund::evidence::Numeric evidence = rund::evidence::make(contract);
  TEST_ASSERT(evidence);
  TEST_ASSERT(evidence.contract().has_value());
  TEST_ASSERT(*evidence.contract() == contract);
  TEST_ASSERT(evidence.id() == rund::evidence::identify(contract));
  const std::string encoded = rund::evidence::encode(evidence);
  const rund::evidence::Numeric decoded = rund::evidence::decode(encoded);
  TEST_ASSERT(decoded);
  TEST_ASSERT(decoded.ok());
  TEST_ASSERT(decoded.code() == rund::evidence::Numeric::Code::Ok);
  TEST_ASSERT(decoded.exit_code() == 0);
  TEST_ASSERT(decoded.contract().has_value());
  TEST_ASSERT(*decoded.contract() == contract);
  TEST_ASSERT(decoded.id() == evidence.id());
  TEST_ASSERT(decoded.strict_float() == evidence.strict_float());
  TEST_ASSERT(decoded.hash() == evidence.hash());
}

} // namespace

int RunEvidenceNumericContract() {
  const rund::evidence::Numeric empty{};
  TEST_ASSERT(!empty);
  TEST_ASSERT(empty.code() == rund::evidence::Numeric::Code::NotBuilt);
  TEST_ASSERT(empty.error() == "numeric_evidence_not_built");
  TEST_ASSERT(empty.exit_code() == 1);
  TEST_ASSERT(rund::evidence::encode(empty).empty());

  const rund::evidence::Numeric evidence =
      rund::evidence::make(rund::evidence::strict_f64());
  TEST_ASSERT(evidence);
  TEST_ASSERT(evidence.code() == rund::evidence::Numeric::Code::Ok);
  TEST_ASSERT(evidence.error().empty());
  TEST_ASSERT(evidence.exit_code() == 0);
  TEST_ASSERT(evidence.id().value == kStrictF64Id);
  TEST_ASSERT(evidence.strict_float());
  TEST_ASSERT(evidence.hash() == kStrictF64Hash);
  TEST_ASSERT(rund::evidence::encode(evidence) == kStrictF64Text);

  const rund::evidence::Numeric invalid =
      rund::evidence::make(rund::evidence::Contract{
          .domain = rund::evidence::Domain::Fixed,
          .arithmetic = rund::evidence::Arithmetic::FixedPoint,
          .authority = rund::evidence::Authority::Authoritative,
          .determinism = rund::evidence::Determinism::Required,
          .integer_bits = 16u,
          .fraction_bits = 15u,
      });
  TEST_ASSERT(!invalid);
  TEST_ASSERT(invalid.code() == rund::evidence::Numeric::Code::BadValue);
  TEST_ASSERT(invalid.error() == "numeric_evidence_bad_value");

  std::string bad_header{kStrictF64Text};
  TEST_ASSERT(replace_line(bad_header, "rund.numeric.evidence",
                           "rund.numeric.evidence.invalid"));
  assert_rejected(bad_header, rund::evidence::Numeric::Code::BadHeader,
                  "numeric_evidence_bad_header");

  std::string duplicate{kStrictF64Text};
  TEST_ASSERT(insert_duplicate_line(duplicate, "schema="));
  assert_rejected(duplicate, rund::evidence::Numeric::Code::DuplicateField,
                  "numeric_evidence_duplicate_field");

  std::string missing{kStrictF64Text};
  TEST_ASSERT(remove_line(missing, "domain="));
  assert_rejected(missing, rund::evidence::Numeric::Code::MissingField,
                  "numeric_evidence_missing_field");

  std::string bad_enum{kStrictF64Text};
  TEST_ASSERT(replace_line(bad_enum, "domain=", "domain=255"));
  assert_rejected(bad_enum, rund::evidence::Numeric::Code::BadValue,
                  "numeric_evidence_bad_value");

  std::string bad_width{kStrictF64Text};
  TEST_ASSERT(replace_line(bad_width, "domain=", "domain=4"));
  TEST_ASSERT(replace_line(bad_width, "arithmetic=", "arithmetic=2"));
  TEST_ASSERT(replace_line(bad_width, "integer_bits=", "integer_bits=16"));
  TEST_ASSERT(replace_line(bad_width, "fraction_bits=", "fraction_bits=15"));
  assert_rejected(bad_width, rund::evidence::Numeric::Code::BadValue,
                  "numeric_evidence_bad_value");

  std::string bad_hash{kStrictF64Text};
  TEST_ASSERT(replace_line(bad_hash, "hash=", "hash=1"));
  assert_rejected(bad_hash, rund::evidence::Numeric::Code::HashInvalid,
                  "numeric_evidence_hash_invalid");

  constexpr rund::evidence::Domain domains[] = {
      rund::evidence::Domain::I32,   rund::evidence::Domain::U32,
      rund::evidence::Domain::I64,   rund::evidence::Domain::U64,
      rund::evidence::Domain::Fixed, rund::evidence::Domain::F32,
      rund::evidence::Domain::F64,
  };
  constexpr rund::evidence::Arithmetic arithmetic_laws[] = {
      rund::evidence::Arithmetic::IntegerWrap,
      rund::evidence::Arithmetic::IntegerSaturating,
      rund::evidence::Arithmetic::FixedPoint,
      rund::evidence::Arithmetic::FloatingPoint,
      rund::evidence::Arithmetic::StrictFloatingPoint,
      rund::evidence::Arithmetic::ReductionSlot,
      rund::evidence::Arithmetic::HashDigest,
  };
  constexpr rund::evidence::Authority authorities[] = {
      rund::evidence::Authority::Authoritative,
      rund::evidence::Authority::Derived,
      rund::evidence::Authority::PresentationOnly,
      rund::evidence::Authority::Diagnostic,
  };
  constexpr rund::evidence::Determinism determinism_policies[] = {
      rund::evidence::Determinism::Required,
      rund::evidence::Determinism::BestEffort,
      rund::evidence::Determinism::NotRequired,
  };
  constexpr rund::evidence::Rounding rounding_policies[] = {
      rund::evidence::Rounding::TowardZero,
      rund::evidence::Rounding::Down,
      rund::evidence::Rounding::Up,
      rund::evidence::Rounding::NearestEven,
  };
  constexpr rund::evidence::Overflow overflow_policies[] = {
      rund::evidence::Overflow::Saturate,
      rund::evidence::Overflow::Wrap,
  };
  constexpr rund::evidence::Approximation approximation_policies[] = {
      rund::evidence::Approximation::Exact,
      rund::evidence::Approximation::Deterministic,
  };

  std::uint64_t valid_count = 0u;
  for (const auto domain : domains) {
    for (const auto arithmetic : arithmetic_laws) {
      for (const auto authority : authorities) {
        for (const auto determinism : determinism_policies) {
          if (domain != rund::evidence::Domain::Fixed) {
            const rund::evidence::Contract contract{
                .domain = domain,
                .arithmetic = arithmetic,
                .authority = authority,
                .determinism = determinism,
            };
            if (rund::evidence::valid(contract)) {
              assert_round_trip(contract);
              ++valid_count;
            }
            continue;
          }
          for (const std::uint8_t width :
               {std::uint8_t{32u}, std::uint8_t{64u}}) {
            for (std::uint8_t integer = 1u; integer < width; ++integer) {
              for (const auto rounding : rounding_policies) {
                for (const auto overflow : overflow_policies) {
                  for (const auto approximation : approximation_policies) {
                    const rund::evidence::Contract contract{
                        .domain = domain,
                        .arithmetic = arithmetic,
                        .authority = authority,
                        .determinism = determinism,
                        .integer_bits = integer,
                        .fraction_bits =
                            static_cast<std::uint8_t>(width - integer),
                        .rounding = rounding,
                        .overflow = overflow,
                        .approximation = approximation,
                    };
                    if (rund::evidence::valid(contract)) {
                      assert_round_trip(contract);
                      ++valid_count;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  // 138 canonical non-fixed contracts plus
  // (31 + 63) fixed splits * 4 roundings * 2 overflows *
  // 2 approximation policies * 10 authority/determinism pairs.
  TEST_ASSERT(valid_count == 15178u);
  return 0;
}
