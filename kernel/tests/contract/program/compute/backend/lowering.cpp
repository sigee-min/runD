namespace program_compute_contract {

int RunComputeBackendLoweringBaseContract();
int RunComputeBackendLoweringMetalContract();
int RunComputeBackendLoweringNameContract();
int RunComputeVulkanLoweringContract();
int RunComputeBackendHelperEmissionContract();
int RunComputeBackendLoweringRejectContract();
int RunComputeBackendLoweringArtifactContract();
int RunComputeBackendLoweringRuntimeContract();

int RunComputeBackendLoweringContract() {
  if (RunComputeBackendLoweringBaseContract() != 0) {
    return 1;
  }
  if (RunComputeBackendLoweringMetalContract() != 0) {
    return 1;
  }
  if (RunComputeBackendLoweringNameContract() != 0) {
    return 1;
  }
  if (RunComputeVulkanLoweringContract() != 0) {
    return 1;
  }
  if (RunComputeBackendHelperEmissionContract() != 0) {
    return 1;
  }
  if (RunComputeBackendLoweringRejectContract() != 0) {
    return 1;
  }
  if (RunComputeBackendLoweringArtifactContract() != 0) {
    return 1;
  }
  return RunComputeBackendLoweringRuntimeContract();
}

} // namespace program_compute_contract
