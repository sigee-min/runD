#include "contract/program/compute/fixed/arithmetic/local.hpp"

namespace program_compute_contract {
namespace {

using namespace lowering_support;

int test_fixed_arithmetic_metal_lowering() {
  const rund::kernel::LoweringArtifact artifact32 =
      rund::kernel::LowerComputeIR(BuildFixedLane32ArithmeticOps().ir(),
                                   rund::kernel::ComputeApi::Metal);
  const rund::kernel::LoweringArtifact artifact64 =
      rund::kernel::LowerComputeIR(BuildFixedLane64ArithmeticOps().ir(),
                                   rund::kernel::ComputeApi::Metal);

  TEST_ASSERT(artifact32.ok);
  TEST_ASSERT(artifact64.ok);
  TEST_ASSERT(artifact32.source_text.find("].op=add_sat") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("].op=add_sat_unsigned") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("].op=sub_sat") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("].op=neg_positive_fixed") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("].op=mul_fixed") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("].op=mul_fixed_scaled") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("].op=mul_unsigned_fixed") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("].op=mul_add_fixed") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("].op=clamp") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("].op=lt") != std::string_view::npos);
  return 0;
}

int test_fixed_arithmetic_vulkan_lowering() {
  const rund::kernel::LoweringArtifact artifact32 =
      rund::kernel::LowerComputeIR(BuildFixedLane32ArithmeticOps().ir(),
                                   rund::kernel::ComputeApi::Vulkan);
  const rund::kernel::LoweringArtifact artifact64 =
      rund::kernel::LowerComputeIR(BuildFixedLane64ArithmeticOps().ir(),
                                   rund::kernel::ComputeApi::Vulkan);

  TEST_ASSERT(artifact32.ok);
  TEST_ASSERT(artifact64.ok);
  TEST_ASSERT(artifact32.source_text.find("].op=add_sat") !=
              std::string_view::npos);
  TEST_ASSERT(artifact64.source_text.find("].op=mul_add_fixed") !=
              std::string_view::npos);
  TEST_ASSERT(artifact64.source_text.find("].op=clamp") !=
              std::string_view::npos);
  TEST_ASSERT(artifact64.source_text.find("].op=lt") != std::string_view::npos);
  return 0;
}

[[nodiscard]] bool HasDeclaredMultiplySource(const std::string &source,
                                             const unsigned fraction,
                                             const unsigned width) {
  const auto has_line = [&](const std::string_view first,
                            const std::string_view second,
                            const std::string_view third = {}) {
    std::size_t begin = 0u;
    while (begin < source.size()) {
      const std::size_t end = source.find('\n', begin);
      const std::string_view line{
          source.data() + begin,
          (end == std::string::npos ? source.size() : end) - begin};
      if (line.find(first) != std::string_view::npos &&
          line.find(second) != std::string_view::npos &&
          (third.empty() || line.find(third) != std::string_view::npos)) {
        return true;
      }
      if (end == std::string::npos) {
        break;
      }
      begin = end + 1u;
    }
    return false;
  };
  const std::string product_fraction = std::to_string(fraction * 2u) + "u";
  const std::string stored_fraction = std::to_string(fraction) + "u";
  const std::string stored_width = std::to_string(width) + "u";
  const std::string signed_quantize =
      product_fraction + ", " + stored_fraction + ", 4u, 1u, " + stored_width;
  return source.find("].op=mul_fixed") != std::string::npos &&
         source.find("].op=mul_fixed_scaled") != std::string::npos &&
         source.find("].op=mul_unsigned_fixed") != std::string::npos &&
         source.find("].op=mul_add_fixed") != std::string::npos &&
         has_line("RundWideQuantize(RundWideMul(", signed_quantize) &&
         has_line("RundWideQuantize(RundWideMul(", "RundWideUnsignedLane(",
                  signed_quantize) &&
         has_line("RundWideQuantizeUnsignedFixedProduct(RundWideMul(",
                  "RundWideUnsignedLane(", signed_quantize) &&
         has_line("RundWideAdd(RundWideMul(", ");");
}

int test_fixed_arithmetic_all_backends_lower_declared_multiply_formats() {
  const auto fixed16 = BuildFixed16_16DeclaredMultiplyOps();
  const auto fixed44 = BuildFixed20_44DeclaredMultiplyOps();
  for (const auto api :
       {rund::kernel::ComputeApi::Metal, rund::kernel::ComputeApi::Vulkan}) {
    const auto artifact16 = rund::kernel::LowerComputeIR(fixed16.ir(), api);
    const auto artifact44 = rund::kernel::LowerComputeIR(fixed44.ir(), api);
    TEST_ASSERT(artifact16.ok);
    TEST_ASSERT(artifact44.ok);
    TEST_ASSERT(HasDeclaredMultiplySource(artifact16.source_text, 16u, 32u));
    TEST_ASSERT(HasDeclaredMultiplySource(artifact44.source_text, 44u, 64u));
  }
  return 0;
}

} // namespace

int RunComputeFixedArithmeticLoweringContract() {
  if (test_fixed_arithmetic_metal_lowering() != 0) {
    return 1;
  }
  if (test_fixed_arithmetic_vulkan_lowering() != 0) {
    return 1;
  }
  return test_fixed_arithmetic_all_backends_lower_declared_multiply_formats();
}

} // namespace program_compute_contract
