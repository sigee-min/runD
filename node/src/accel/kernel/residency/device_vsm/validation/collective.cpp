#include "proof.hpp"

namespace rund::node::accel::detail {

bool device_vsm_scan_proof_valid(const DeviceVsmProof &proof) noexcept {
  const DeviceVsmScanProof &scan = proof.scan;
  const rund::kernel::ScanPlan &semantic = scan.semantic;
  const std::uint64_t element = proof.geometry.element_bytes;
  const bool inclusive = semantic.op == rund::kernel::ScanOp::InclusiveSum;
  const bool exclusive = semantic.op == rund::kernel::ScanOp::ExclusiveSum;
  const bool valid_map =
      scan.map.kind == DeviceVsmScanMapKind::None ||
      (semantic.element == rund::kernel::ScanElement::U64 &&
       (scan.map.kind == DeviceVsmScanMapKind::AddWrapU64Immediate ||
        scan.map.kind == DeviceVsmScanMapKind::CanonicalTotalU64));
  const bool canonical_map =
      scan.map.kind == DeviceVsmScanMapKind::CanonicalTotalU64;
  const bool geometry =
      inclusive ? device_vsm_complete_frame_geometry(proof.geometry)
                : proof.geometry.read_prefix_bytes == element &&
                      proof.geometry.target_offset_bytes == element &&
                      proof.geometry.read_suffix_bytes == 0u &&
                      proof.geometry.frame_bytes ==
                          proof.geometry.payload_bytes + element;
  return proof.topology == DeviceVsmTopology::Scan && semantic.ok && geometry &&
         valid_map && (inclusive || exclusive) &&
         (semantic.element == rund::kernel::ScanElement::U32 ||
          semantic.element == rund::kernel::ScanElement::U64) &&
         semantic.element_bytes ==
             (semantic.element == rund::kernel::ScanElement::U32
                  ? sizeof(std::uint32_t)
                  : sizeof(std::uint64_t)) &&
         semantic.element_bytes == element &&
         semantic.element_count == proof.geometry.frame_bytes / element &&
         semantic.count_source == rund::kernel::ComputeCountSource::Descriptor &&
         scan.workgroup_width == 256u &&
         scan.stage_count ==
             1u + static_cast<std::uint32_t>(scan.map.active()) &&
         proof.output_bytes == proof.geometry.logical_bytes &&
         proof.parameter_bytes == proof.plan.param_bytes &&
         (canonical_map || proof.parameter_bytes == 0u) &&
         proof.plan.input_buffer_count ==
             (canonical_map ? proof.residents.input_count : 1u) &&
         proof.plan.output_buffer_count == 1u &&
         proof.plan.dispatch_count == 1u;
}

bool device_vsm_reduce_proof_valid(const DeviceVsmProof &proof) noexcept {
  const DeviceVsmReduceProof &reduce = proof.reduce;
  const rund::kernel::ReducePlan &semantic = reduce.semantic;
  const std::uint64_t element = proof.geometry.element_bytes;
  const bool operation = semantic.op == rund::kernel::ReduceOp::Sum ||
                         semantic.op == rund::kernel::ReduceOp::CountNonzero ||
                         semantic.op == rund::kernel::ReduceOp::Min ||
                         semantic.op == rund::kernel::ReduceOp::Max;
  return proof.topology == DeviceVsmTopology::Reduce && semantic.ok &&
         operation &&
         (semantic.element == rund::kernel::ReduceElement::U32 ||
          semantic.element == rund::kernel::ReduceElement::U64) &&
         semantic.element_bytes ==
             (semantic.element == rund::kernel::ReduceElement::U32
                  ? sizeof(std::uint32_t)
                  : sizeof(std::uint64_t)) &&
         semantic.element_bytes == element &&
         semantic.element_count == proof.geometry.frame_bytes / element &&
         semantic.count_source == rund::kernel::ComputeCountSource::Descriptor &&
         reduce.workgroup_width == 256u &&
         device_vsm_complete_frame_geometry(proof.geometry) &&
         proof.output_bytes == element && proof.parameter_bytes == 0u &&
         proof.plan.param_bytes == 0u && proof.plan.input_buffer_count == 1u &&
         proof.plan.output_buffer_count == 1u &&
         proof.plan.dispatch_count == 1u;
}

} // namespace rund::node::accel::detail
