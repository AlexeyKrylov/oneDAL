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

#include <daal/src/algorithms/svm/svm_train_thunder_kernel.h>

#include "oneapi/dal/algo/svm/backend/cpu/train_kernel.hpp"
#include "oneapi/dal/algo/svm/backend/interop_model.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

namespace oneapi::dal::svm::backend {

using std::int64_t;
using dal::backend::context_cpu;

namespace daal_svm             = daal::algorithms::svm;
namespace daal_kernel_function = daal::algorithms::kernel_function;
namespace interop              = dal::backend::interop;

template <typename Float, daal::CpuType Cpu>
using daal_svm_thunder_kernel_t = daal_svm::training::internal::
    SVMTrainImpl<daal_svm::training::thunder, Float, daal_svm::Parameter, Cpu>;

template <typename Float>
static train_result call_daal_kernel(const context_cpu& ctx,
                                     const descriptor_base& desc,
                                     const table& data,
                                     const table& labels,
                                     const table& weights) {
    const int64_t row_count    = data.get_row_count();
    const int64_t column_count = data.get_column_count();

    // TODO: data is table, not a homogen_table. Think better about accessor - is it enough to have just a row_accessor?
    auto arr_data    = row_accessor<const Float>{ data }.pull();
    auto arr_label   = row_accessor<const Float>{ labels }.pull();
    auto arr_weights = row_accessor<const Float>{ weights }.pull();

    const auto daal_data =
        interop::convert_to_daal_homogen_table(arr_data, row_count, column_count);
    const auto daal_labels  = interop::convert_to_daal_homogen_table(arr_label, row_count, 1);
    const auto daal_weights = interop::convert_to_daal_homogen_table(arr_weights, row_count, 1);

    // TODO: move as parameter onedal SVM
    auto kernel =
        daal_kernel_function::KernelIfacePtr(new daal_kernel_function::linear::Batch<Float>());

    daal_svm::Parameter daal_parameter(
        kernel,
        desc.get_c(),
        desc.get_accuracy_threshold(),
        desc.get_tau(),
        desc.get_max_iteration_count(),
        int64_t(desc.get_cache_size() * 1024 * 1024), // DAAL get in bytes
        desc.get_shrinking());

    auto daal_model = daal_svm::Model::create<Float>(column_count);
    interop::call_daal_kernel<Float, daal_svm_thunder_kernel_t>(ctx,
                                                                daal_data,
                                                                daal_weights,
                                                                *daal_labels,
                                                                daal_model.get(),
                                                                &daal_parameter);
    auto table_support_indices =
        interop::convert_from_daal_homogen_table<Float>(daal_model->getSupportIndices());

    return train_result()
        .set_model(convert_from_daal_model<Float>(*daal_model))
        .set_support_indices(table_support_indices);
}

template <typename Float>
static train_result train(const context_cpu& ctx,
                          const descriptor_base& desc,
                          const train_input& input) {
    return call_daal_kernel<Float>(ctx,
                                   desc,
                                   input.get_data(),
                                   input.get_labels(),
                                   input.get_weights());
}

template <typename Float>
struct train_kernel_cpu<Float, task::classification, method::thunder> {
    train_result operator()(const context_cpu& ctx,
                            const descriptor_base& desc,
                            const train_input& input) const {
        return train<Float>(ctx, desc, input);
    }
};

template struct train_kernel_cpu<float, task::classification, method::thunder>;
template struct train_kernel_cpu<double, task::classification, method::thunder>;

} // namespace oneapi::dal::svm::backend
