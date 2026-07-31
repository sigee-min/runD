#include "local.hpp"
#include <node/accel/cpu/simd.hpp>

#include <array>
#include <string_view>

namespace node_accel_contract {

[[nodiscard]] bool RejectsForgedArtifactWithoutWritingOutput() {
  constexpr std::size_t kTileCount = 4u;
  std::array<rund::kernel::i32, kTileCount> input{1, 2, 3, 4};
  std::array<rund::kernel::i32, kTileCount> out{91, 92, 93, 94};

  const auto body = rund::compute_dsl::bind(kTileCount)
                        .fixed<1, 31>()
                        .read<"input">(input.data())
                        .write<"out">(out.data());
  const rund::compute_dsl::ComputeOp op =
      rund::compute_dsl::def("node-cpu-simd-forged")
          .on(body)
          .map([](auto i, auto b) {
            auto input = b.template read<"input">();
            auto out = b.template write<"out">();
            out[i] = input[i] + 1;
          });

  const rund::kernel::CpuCaps caps = cpu::NeonCaps();
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Cpu);
  const rund::kernel::BindingSet bindings = op.bindings<rund::kernel::i32>(
      19u, caps.fixed_lane32_lanes, rund::kernel::ComputeApi::Metal);

  rund::kernel::CpuCaps invalid_caps = caps;
  ++invalid_caps.lane_bytes;
  const auto invalid =
      rund::node::accel::RunCpuSimd(op.ir(), invalid_caps, artifact, bindings);
  TEST_ASSERT(!invalid.ok);
  TEST_ASSERT(std::string_view{invalid.reason} == "cpu_caps_invalid");

  rund::kernel::LoweringArtifact canonical_forged = artifact;
  canonical_forged.canonical_ir_bytes.push_back(0xffu);
  const auto canonical =
      rund::node::accel::RunCpuSimd(op.ir(), caps, canonical_forged, bindings);
  TEST_ASSERT(!canonical.ok);
  TEST_ASSERT(std::string_view{canonical.reason} ==
              "compute_artifact_mismatch");

  rund::kernel::LoweringArtifact forged = artifact;
  forged.source_text += "\nforged";
  const rund::node::accel::CpuSimdRunResult run =
      rund::node::accel::RunCpuSimd(op.ir(), caps, forged, bindings);

  TEST_ASSERT(!run.ok);
  TEST_ASSERT(std::string_view{run.reason} == "compute_artifact_mismatch");
  TEST_ASSERT(run.rejected_count == 1u);
  TEST_ASSERT(out[0] == 91);
  TEST_ASSERT(out[1] == 92);
  TEST_ASSERT(out[2] == 93);
  TEST_ASSERT(out[3] == 94);
  return true;
}

} // namespace node_accel_contract
