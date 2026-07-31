namespace program_compute_contract {

int RunComputeDslContract();
int RunFusionContract();
int RunComputeFixedNonlinearFusionContract();
int RunGraphContract();
int RunComputeIrContract();
int RunComputeBackendLoweringContract();
int RunComputeMetadataContract();
int RunComputePlanContract();
int RunCompactContract();
int RunGatherContract();
int RunHistogramContract();
int RunPartitionContract();
int RunReduceContract();
int RunScatterContract();
int RunScatterReduceContract();
int RunScanContract();
int RunSegmentedScanContract();
int RunSegmentedReduceContract();
int RunSortContract();
int RunStencilContract();
int RunNumericAlgebraContract();
int RunComputeFixedArithmeticContract();

} // namespace program_compute_contract

int RunComputeModelContract() {
  if (const int rc = program_compute_contract::RunComputePlanContract(); rc != 0) {
    return rc;
  }
  if (const int rc = program_compute_contract::RunGraphContract(); rc != 0) {
    return rc;
  }
  if (const int rc = program_compute_contract::RunFusionContract(); rc != 0) {
    return rc;
  }
  if (const int rc = program_compute_contract::RunScanContract(); rc != 0) {
    return rc;
  }
  if (const int rc = program_compute_contract::RunSegmentedScanContract();
      rc != 0) {
    return rc;
  }
  if (const int rc = program_compute_contract::RunSegmentedReduceContract();
      rc != 0) {
    return rc;
  }
  if (const int rc = program_compute_contract::RunSortContract(); rc != 0) {
    return rc;
  }
  if (const int rc = program_compute_contract::RunStencilContract();
      rc != 0) {
    return rc;
  }
  if (const int rc = program_compute_contract::RunCompactContract(); rc != 0) {
    return rc;
  }
  if (const int rc = program_compute_contract::RunGatherContract(); rc != 0) {
    return rc;
  }
  if (const int rc = program_compute_contract::RunHistogramContract();
      rc != 0) {
    return rc;
  }
  if (const int rc = program_compute_contract::RunPartitionContract(); rc != 0) {
    return rc;
  }
  if (const int rc = program_compute_contract::RunReduceContract(); rc != 0) {
    return rc;
  }
  if (const int rc = program_compute_contract::RunScatterContract(); rc != 0) {
    return rc;
  }
  if (const int rc = program_compute_contract::RunScatterReduceContract();
      rc != 0) {
    return rc;
  }
  if (const int rc = program_compute_contract::RunNumericAlgebraContract();
      rc != 0) {
    return rc;
  }
  if (const int rc = program_compute_contract::RunComputeFixedNonlinearFusionContract();
      rc != 0) {
    return rc;
  }
  if (const int rc = program_compute_contract::RunComputeIrContract(); rc != 0) {
    return rc;
  }
  if (const int rc = program_compute_contract::RunComputeDslContract(); rc != 0) {
    return rc;
  }
  if (const int rc = program_compute_contract::RunComputeBackendLoweringContract();
      rc != 0) {
    return rc;
  }
  if (const int rc = program_compute_contract::RunComputeFixedArithmeticContract(); rc != 0) {
    return rc;
  }
  if (const int rc = program_compute_contract::RunComputeMetadataContract(); rc != 0) {
    return rc;
  }
  return 0;
}
