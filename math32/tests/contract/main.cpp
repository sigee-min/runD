int RunMath32PublicContract();
int RunMath32SimdContract();
int RunMath32FixedContract();
int RunMath32NonlinearContract();
int RunMath32TurnContract();
int RunMath32StatContract();
int RunMath32GeometryVectorContract();
int RunMath32SoaContract();
int RunMath32QuantContract();
int RunMath32NnContract();
int RunMath32ProbContract();
int RunMath32RlContract();

int main() {
  RunMath32PublicContract();
  RunMath32SimdContract();
  RunMath32FixedContract();
  RunMath32NonlinearContract();
  RunMath32TurnContract();
  RunMath32StatContract();
  RunMath32GeometryVectorContract();
  RunMath32SoaContract();
  RunMath32QuantContract();
  RunMath32NnContract();
  RunMath32ProbContract();
  RunMath32RlContract();
  return 0;
}
