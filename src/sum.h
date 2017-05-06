#include <stdint.h>
#include "immintrin.h"

bool IsAligned(size_t alignment, void* pointer);

float Sum(const float* data, int32_t size);

__m128 PartialSumsSse(const float* data, int32_t size);

__m128d PartialSumsSse(const double* data, int32_t size);

__m256 PartialSumsAvx(const float* data, int32_t size);

float HorizontalSumSse3(__m128 a);

float HorizontalSum(__m128 a);

double HorizontalSumSse3(__m128d a);

float HorizontalSum(__m256 a);

float SumSse(const float* data, int32_t size);

double SumSse(const double* data, int32_t size);

float SumAvx(const float* data, int32_t size);

double MeanSse(const double* data, int32_t size);

float MeanAvx(const float* data, int32_t size);

struct MinMax
{
  float Min;
  float Max;
};

MinMax ComputeMinMax(const float* data, int32_t size);

MinMax ComputeMinMaxSse(const float* data, int32_t size);

