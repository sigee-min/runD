namespace program_compute_contract {

int RunComputeDslIdentityContract();
int RunComputeDslCseContract();
int RunComputeDslOpsContract();
int RunComputeDslRejectContract();
int RunComputeDslEscapeContract();
int RunComputeDslShapeContract();
int RunComputeDslPlanContract();

int RunComputeDslContract() {
  if (RunComputeDslIdentityContract() != 0) {
    return 1;
  }
  if (RunComputeDslCseContract() != 0) {
    return 1;
  }
  if (RunComputeDslOpsContract() != 0) {
    return 1;
  }
  if (RunComputeDslRejectContract() != 0) {
    return 1;
  }
  if (RunComputeDslEscapeContract() != 0) {
    return 1;
  }
  if (RunComputeDslShapeContract() != 0) {
    return 1;
  }
  return RunComputeDslPlanContract();
}

}  // namespace program_compute_contract
