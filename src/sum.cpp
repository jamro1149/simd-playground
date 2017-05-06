#include "sum.h"
#include "emmintrin.h"
#include "xmmintrin.h"
#include "pmmintrin.h"
#include "immintrin.h"
#include <stdio.h>
#include <assert.h>

bool IsAligned(const size_t alignment, const void* pointer)
{
	const uintptr_t ptrAsInt = reinterpret_cast<uintptr_t>(pointer);
	
	return ptrAsInt % (alignment) == 0;
}

float Sum(const float* data, const int32_t size)
{
  float res = 0.f;

  for (int32_t i = 0; i < size; ++i)
  {
    res += data[i];
  }

  return res;
}

float HorizontalSumSse3(__m128 a)
{
  float ret;

  a = _mm_hadd_ps(a, a);
  a = _mm_hadd_ps(a, a);

  _mm_store_ss(&ret, a);

  return ret;
}

double HorizontalSumSse3(__m128d a)
{
  double ret;

  a = _mm_hadd_pd(a, __m128d());

  _mm_store_sd(&ret, a);

  return ret;
}

float HorizontalSum(__m128 a)
{
  float ret;

  // partialAdd[0] = a[0] + a[1]
  // partialAdd[2] = a[2] + a[3]
  // don't care about other lanes
  __m128 shuffled = _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 3, 0, 1));
  __m128 partialAdd = _mm_add_ps(a, shuffled);

  // ret[0] = partialAdd[0] + partialAdd[2]
  // don't care about other lanes
  __m128 shuffledAgain =
    _mm_shuffle_ps(partialAdd, partialAdd, _MM_SHUFFLE(0, 0, 0, 2));
  __m128 Sum = _mm_add_ps(partialAdd, shuffledAgain);
  
  _mm_store_ss(&ret, Sum);

  return ret;
}

float HorizontalSum(__m256 a)
{
  float ret;

  __m256 PartialSums = _mm256_hadd_ps(a, __m256());
  PartialSums = _mm256_hadd_ps(PartialSums, __m256());
  __m128 PartialSumHigher = _mm256_extractf128_ps(PartialSums, 1);
  __m128 PartialSumLower = _mm256_castps256_ps128(PartialSums);
  __m128 Sum = _mm_add_ps(PartialSumHigher, PartialSumLower);

  _mm_store_ss(&ret, Sum);
  return ret;
}

__m128 PartialSumsSse(const float* data, const int32_t size)
{
  assert(IsAligned(16, data));
  
  const int32_t NumStragglers = size % 4;
  const int32_t SizeWithoutStragglers = size - NumStragglers;

  __m128 PartialSums = _mm_set_ps1(0);

  for (int32_t i = 0; i < SizeWithoutStragglers; i += 4)
  {
    __m128 FourFloats = _mm_load_ps(data + i);

    PartialSums = _mm_add_ps(PartialSums, FourFloats);
  }

  for (int32_t i = SizeWithoutStragglers; i < size; ++i)
  {
    __m128 Straggler = _mm_set_ps1(data[i]);
    PartialSums = _mm_add_ss(PartialSums, Straggler);
  }

  return PartialSums;
}

__m128d PartialSumsSse(const double* data, const int32_t size)
{
  assert(IsAligned(16, data));
  
  const int32_t NumStragglers = size % 2;
  const int32_t SizeWithoutStragglers = size - NumStragglers;

  __m128d PartialSums = _mm_set_pd(0, 0);

  for (int32_t i = 0; i < SizeWithoutStragglers; i += 2)
  {
    __m128d TwoDoubles = _mm_load_pd(data + i);

    PartialSums = _mm_add_pd(PartialSums, TwoDoubles);
  }

  for (int32_t i = SizeWithoutStragglers; i < size; ++i)
  {
    __m128d Straggler = _mm_set_pd(0, data[i]);
    PartialSums = _mm_add_sd(PartialSums, Straggler);
  }

  return PartialSums;
}

__m256 PartialSumsAvx(const float* data, const int32_t size)
{
  assert(IsAligned(32, data));

  const int32_t NumStragglers = size % 8;
  const int32_t SizeWithoutStragglers = size - NumStragglers;

  __m256 PartialSums;
  PartialSums = _mm256_set_ps(0, 0, 0, 0, 0, 0, 0, 0);

  for (int32_t i = 0; i < SizeWithoutStragglers; i += 8)
  {
    __m256 EightFloats = _mm256_load_ps(data + i);

    PartialSums = _mm256_add_ps(PartialSums, EightFloats);
  }

  for (int32_t i = SizeWithoutStragglers; i < size; ++i)
  {
    __m256 Straggler = _mm256_set_ps(0, 0, 0, 0, 0, 0, 0, data[i]);
    PartialSums = _mm256_add_ps(PartialSums, Straggler);
  }

  return PartialSums;
}

float SumSse(const float* data, const int32_t size)
{
  const __m128 PartialSums = PartialSumsSse(data, size);
  return HorizontalSumSse3(PartialSums);
}

double SumSse(const double* data, const int32_t size)
{
  const __m128d PartialSums = PartialSumsSse(data, size);
  return HorizontalSumSse3(PartialSums);
}

float SumAvx(const float* data, const int32_t size)
{
  const __m256 PartialSums = PartialSumsAvx(data, size);
  return HorizontalSum(PartialSums);
}

double MeanSse(const double* data, const int32_t size)
{
  assert(size != 0);
  const float Sum = SumSse(data, size);
  return Sum / size;
}

float MeanAvx(const float* data, const int32_t size)
{
  assert(size != 0);
  const float Sum = SumAvx(data, size);
  return Sum / size;
}

MinMax ComputeMinMax(const float* data, const int32_t size)
{
  MinMax ret;

  assert(size > 0);

  ret.Min = data[0];
  ret.Max = data[0];

  for (int32_t i = 1; i + 1 < size; i += 2)
  {
    const float left = data[i];
    const float right = data[i + 1];
    const bool leftSmaller = left < right;
    const float smaller = leftSmaller ? left : right;
    const float larger = leftSmaller ? right : left;
    if (smaller < ret.Min)
    {
      ret.Min = smaller;
    }
    else if (larger > ret.Max)
    {
      ret.Max = larger;
    }
  }

  if (size % 2 == 0)
  {
    const float last = data[size - 1];
    if (last < ret.Min)
    {
      ret.Min = last;
    }
    else if (last > ret.Max)
    {
      ret.Max = last;
    }
  }

  return ret;
}

