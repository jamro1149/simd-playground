#include <algorithm>
#include <chrono>
#include <random>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include "immintrin.h"
#include "sum.h"

void* AlignedAllocate(int32_t alignment, size_t size)
{
#ifdef _WIN32
  return _aligned_malloc(alignment, size);
#else
  return aligned_alloc(alignment, size);
#endif
}

float UniformFloat(float min, float max)
{
  int randomNum = rand();
  float cannonicalFloat = (RAND_MAX - randomNum) / (float)RAND_MAX;
  float range = max - min;
  float zeroToRange = cannonicalFloat * range;
  return zeroToRange + min;
}

float* Generate(int32_t num, float min, float max, uint32_t seed)
{
  float* ret = static_cast<float*>((AlignedAllocate(32, num * sizeof(float))));

  srand(seed);

  // would love to use these, but it crashes on clang
  //std::mt19937 gen(seed);
  //std::normal_distribution<float> dis(min, max);

  for (int32_t i = 0; i < num; ++i)
  {
    float newFloat = UniformFloat(min, max);
    ret[i] = newFloat;
  }

  return ret;
}

template <typename Functor>
double* RunTests_Warm(Functor&& fun, int32_t NumTests)
{
  double* ret = (double*)AlignedAllocate(16, NumTests * sizeof(double));

  // this serves two purposes
  // firstly, it warms up the cache of fun
  // secondly, we can sanity check all future runs of fun
  // to see if we get the same answer
  float expected = fun();

  for (int32_t i = 0; i < NumTests; ++i)
  {
    auto start = std::chrono::high_resolution_clock::now();
    float result = fun();
    auto end = std::chrono::high_resolution_clock::now();

    if (result != expected)
    {
       printf("Error while running tests:\nExpected:\t%f\nActual:\t%f\n",
           expected, result);
       free(ret);
       return nullptr;
    }

    std::chrono::duration<double> runtime = end - start;
    ret[i] = runtime.count();
  }

  return ret;
}

template <typename Functor>
void TestAndPrintResults(Functor&& fun, int32_t NumTests, const char* Name)
{
  printf("Running %s %d times\n", Name, NumTests);

  double* results = RunTests_Warm(fun, NumTests);

  const double Mean = MeanSse(results, NumTests);
  const auto MinAndMax = std::minmax_element(results, results + NumTests);

  printf("Min:\t%f s\n", *MinAndMax.first);
  printf("Mean:\t%f s\n", Mean);
  printf("Max:\t%f s\n", *MinAndMax.second);

  printf("\n");
}

int main(int argc, char** argv)
{
  if (argc != 4)
  {
    printf("Usage: %s [NumFloats] [NumRepetitions] [Seed]\n", argv[0]);
    return 1;
  }

  int32_t NumFloats = strtol(argv[1], nullptr, 10);
  if (errno == ERANGE)
  {
    printf("Error: could not parse '%s' as an integer\n", argv[1]);
    return 2;
  }

  int32_t NumTests = strtol(argv[2], nullptr, 10);
  if (errno == ERANGE)
  {
    printf("Error: could not parse '%s' as an integer\n", argv[2]);
    return 3;
  }
  
  int32_t Seed = strtol(argv[3], nullptr, 10);
  if (errno == ERANGE)
  {
    printf("Error: could not parse '%s' as an integer\n", argv[3]);
    return 4;
  }

  float* floats = Generate(NumFloats, -1.f, 1.f, Seed);

  printf("Generated %d floats using seed %d, including:\n", NumFloats, Seed);
  for (int32_t i = 0; i < fmin(NumFloats, 10); ++i)
  {
    printf("%f\n", floats[i]);
  }
  printf("\n");

  TestAndPrintResults([=](){ return Sum(floats, NumFloats); },
      NumTests, "Scalar Sum");

  TestAndPrintResults([=](){ return SumSse(floats, NumFloats); },
      NumTests, "SSE Sum");

  TestAndPrintResults([=](){ return SumAvx(floats, NumFloats); },
      NumTests, "AVX Sum");

}

