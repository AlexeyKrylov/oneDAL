/** file distances.cpp */
/*******************************************************************************
* Copyright 2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "algorithms/distances.h"
#include "data_management/data/internal/train_test_split.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/data_dictionary.h"
#include "services/env_detect.h"
#include "src/externals/service_dispatch.h"
#include "src/services/service_data_utils.h"
#include "src/services/service_utils.h"
#include "src/services/service_defines.h"
#include "src/threading/threading.h"
#include "src/data_management/service_numeric_table.h"
#include "data_management/features/defines.h"
#include "src/algorithms/service_error_handling.h"
#include "src/externals/service_math_mkl.h"
#include "data_management/data/data_dictionary.h"
#include "data_management/features/defines.h"
#include "immintrin.h"


using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace distances
{

template <typename algorithmFPType, daal::CpuType cpu>
algorithmFPType computeDistance(const algorithmFPType * x, const algorithmFPType * y, const size_t p, const size_t n)
{
    daal::internal::mkl::MklMath<algorithmFPType, cpu> math;

    algorithmFPType d = 0;
    
    switch (p)
    {
    case 0:

        for (size_t i = 0; i < n; ++i)
        {
            if (math.sFabs(x[i] - y[i]) > d)
            {
                d = math.sFabs(x[i] - y[i]);
            }
        }

        return d;

    case 1:

        for (size_t i = 0; i < n; ++i)
        {
            d += math.sFabs(x[i] - y[i]);
        }

        return d;

    default:

        for (size_t i = 0; i < n; ++i)
        {
            d += math.sPowx(math.sFabs(x[i] - y[i]), p);
        }

        return math.sPowx(d, 1.0 / p);
    }
}


#if defined(__INTEL_COMPILER)

template <>
float computeDistance<float, avx512>(const float * x, const float * y, const size_t p, const size_t n)
{
    daal::internal::mkl::MklMath<float, avx512> math;
    float* tmp = new float[16];
    float d = 0.0;

    __m512 * ptr512x = (__m512 *)x;
    __m512 * ptr512y = (__m512 *)y;

    switch (p)
    {
    case 0:

        __m512  tmp512 = _mm512_abs_ps(_mm512_sub_ps(ptr512x[0], ptr512y[0]));
        

        for (size_t i = 1; i < n / 16; ++i)
        {
            tmp512 = _mm512_max_ps(tmp512, _mm512_abs_ps(_mm512_sub_ps(ptr512x[i], ptr512y[i])));
        }

        _mm512_storeu_ps(tmp, tmp512);

        for (size_t i = 0; i < 16; ++i)
        {
            if (tmp[i] > d)
            {
                d = tmp[i];
            }
        }

        delete[] tmp;

        for (size_t i = (n / 16) * 16; i < n; ++i)
        {
            if (math.sFabs(x[i] - y[i]) > d)
            {
                d = math.sFabs(x[i] - y[i]);
            }
        }

        return d;

    case 1:

        for (size_t i = 0; i < n / 16; ++i)
        {
            d += _mm512_reduce_add_ps(_mm512_abs_ps(_mm512_sub_ps(ptr512x[i], ptr512y[i])));
        }

        for (size_t i = (n / 16) * 16; i < n; ++i)
        {
            d += math.sFabs(x[i] - y[i]);
        }

        return d;

    default:

        for (size_t i = 0; i < n / 16 ; ++i)
        {
            _mm512_storeu_ps(tmp, _mm512_abs_ps(_mm512_sub_ps(ptr512x[i], ptr512y[i])));
            math.vPowx(4, tmp, p, tmp);
            d += _mm512_reduce_add_ps(_mm512_loadu_ps(tmp));
        }

        delete[] tmp;

        for (size_t i = (n / 16) * 16; i < n; ++i)
        {
            d += math.sPowx(math.sFabs(x[i] - y[i]), p);
        }

        return math.sPowx(d, 1.0 / p);
    }
}

template <>
double computeDistance<double, avx512>(const double * x, const double * y, const size_t p, const size_t n)
{
    daal::internal::mkl::MklMath<double, avx512> math;

    double d = 0.0;
    double* tmp = new double[8];
    __m512d* ptr512x = (__m512d*)x;
    __m512d* ptr512y = (__m512d*)y;

    switch (p)
    {
    case 0:

        __m512d  tmp512 = _mm512_abs_pd(_mm512_sub_pd(ptr512x[0], ptr512y[0]));

        for (size_t i = 1; i < n / 8; ++i)
        {
            tmp512 = _mm512_max_pd(tmp512, _mm512_abs_pd(_mm512_sub_pd(ptr512x[i], ptr512y[i])));
        }

        _mm512_storeu_pd(tmp, tmp512);

        for (size_t i = 0; i < 8; ++i)
        {
            if (tmp[i] > d)
            {
                d = tmp[i];
            }
        }

        delete[] tmp;

        for (size_t i = (n / 8) * 8; i < n; ++i)
        {
            if (math.sFabs(x[i] - y[i]) > d)
            {
                d = math.sFabs(x[i] - y[i]);
            }
        }

        return d;

    case 1:

        for (size_t i = 0; i < n / 8; ++i)
        {
            d += _mm512_reduce_add_pd(_mm512_abs_pd(_mm512_sub_pd(ptr512x[i], ptr512y[i])));
        }

        for (size_t i = (n / 8) * 8; i < n; ++i)
        {
            d += math.sFabs(x[i] - y[i]);
        }

        return d;

    default:


        for (size_t i = 0; i < n / 8; ++i)
        {
            _mm512_storeu_pd(tmp, _mm512_abs_pd(_mm512_sub_pd(ptr512x[i], ptr512y[i])));
            math.vPowx(8, tmp, p, tmp);
            d += _mm512_reduce_add_pd(_mm512_loadu_pd(tmp));
        }

        delete[] tmp;

        for (size_t i = (n / 8) * 8; i < n; ++i)
        {
            d += math.sPowx(math.sFabs(x[i] - y[i]), p);
        }

        return math.sPowx(d, 1.0 / p);
    }
}

#endif



template <typename algorithmFPType, daal::CpuType cpu>
services::Status minkowskiImpl(const NumericTablePtr & xTable, const NumericTablePtr & yTable,
                               const NumericTablePtr & distancesTable, const size_t & p)
{
        daal::internal::mkl::MklMath<algorithmFPType, cpu> math;

        const size_t nDims = xTable->getNumberOfColumns();
        const size_t nX = xTable->getNumberOfRows();
        const size_t nY = xTable->getNumberOfRows();

        daal::internal::ReadRows<algorithmFPType, cpu> xBlock(*xTable, 0, nX);
        const algorithmFPType* x = xBlock.get();
        DAAL_CHECK_MALLOC(x);

        daal::internal::ReadRows<algorithmFPType, cpu> yBlock(*yTable, 0, nY);
        const algorithmFPType* y = yBlock.get();
        DAAL_CHECK_MALLOC(y);

        daal::internal::WriteRows<algorithmFPType, cpu> distancesBlock(*distancesTable, 0, nX);
        algorithmFPType* distances = distancesBlock.get();
        DAAL_CHECK_MALLOC(distances);

        const size_t BlockSize = 64;
        const size_t THREADING_BORDER = 32768;
        const size_t nBlocksX = nX / BlockSize;
        const size_t nBlocksY = nY / BlockSize;
        const size_t nThreads = threader_get_threads_number();

        if (nThreads > 1 && nX * nY > THREADING_BORDER)
        {
            daal::threader_for(nBlocksX, nBlocksX, [&](size_t iBlockX) {

                const size_t startX = iBlockX * BlockSize;
                const size_t endX = (nBlocksX - iBlockX - 1) ? startX + BlockSize : nX;

                daal::threader_for(nBlocksY, nBlocksY, [&](size_t iBlockY) {

                    const size_t startY = iBlockY * BlockSize;
                    const size_t endY = (nBlocksY - iBlockY - 1) ? startY + BlockSize : nY;

                    for (size_t ix = startX; ix < endX; ++ix)
                        for (size_t iy = startY; iy < endY; ++iy)
                        {
                            distances[ix * nY + iy] = computeDistance<algorithmFPType, cpu>(x + ix * nDims, y + iy * nDims, p, nDims); 
                        }
                });
            });
        }
        else
        {
            for (size_t ix = 0; ix < nX; ++ix)
                for (size_t iy = 0; iy < nY; ++iy)
                {
                    distances[ix * nY + iy] = computeDistance<algorithmFPType, cpu>(x + ix * nDims, y + iy * nDims, p, nDims);
                }
        }

        return services::Status();

    }

template <typename algorithmFPType>
void minkowskiDispImpl(const NumericTablePtr & xTable, const NumericTablePtr & yTable,
                       const NumericTablePtr & distancesTable, const size_t & p)
{
#define DAAL_MINKOWSKI(cpuId, ...) minkowskiImpl<algorithmFPType, cpuId>(__VA_ARGS__);
    DAAL_DISPATCH_FUNCTION_BY_CPU(DAAL_MINKOWSKI, xTable, yTable, distancesTable, p);
#undef DAAL_MINKOWSKI
}

template <typename algorithmFPType>
DAAL_EXPORT void minkowski(const NumericTablePtr & xTable, const NumericTablePtr & yTable,
                           const NumericTablePtr & distancesTable, const size_t & p)
{
    NumericTableDictionaryPtr tableFeaturesDict = xTable->getDictionarySharedPtr();

    switch ((*tableFeaturesDict)[0].getIndexType())
    {
    case daal::data_management::features::IndexNumType::DAAL_FLOAT32:
        DAAL_SAFE_CPU_CALL((minkowskiDispImpl<float>(xTable, yTable, distancesTable, p)),
                           (minkowskiImpl<float, daal::CpuType::sse2>(xTable, yTable, distancesTable, p)));
        break;
    case daal::data_management::features::IndexNumType::DAAL_FLOAT64:
        DAAL_SAFE_CPU_CALL((minkowskiDispImpl<double>(xTable, yTable, distancesTable, p)),
                           (minkowskiImpl<double, daal::CpuType::sse2>(xTable, yTable, distancesTable, p)));
        break;
    }
}

template DAAL_EXPORT void minkowski<float>(const NumericTablePtr & xTable, const NumericTablePtr & yTable,
                                           const NumericTablePtr & distancesTable, const size_t & p);

template DAAL_EXPORT void minkowski<double>(const NumericTablePtr & xTable, const NumericTablePtr & yTable,
                                            const NumericTablePtr & distancesTable, const size_t & p);

} // namespace internal
} // namespace data_management
} // namespace daal
