#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#define logme(x) std::cout << "[pybind] " << x << std::endl;

extern void launch_gather(
    uint32_t blockDim,// void* l2ctrl, 
    void* stream,
    uint8_t* input, int64_t dim, uint8_t* index, uint8_t* output, uint32_t totalLength
);

namespace my_add {
at::Tensor run_gather_custom(const at::Tensor &input, int64_t dim, const at::Tensor &index)
{
    // TORCH_CHECK( index.dim() == 2, " index must be 2 ", index.dim() );
    constexpr int64_t MAX_AI_CORE = 8;
    constexpr int64_t UB_MIN_RD_BLOCK_SIZE = 32;
    constexpr int64_t UB_SIZE = 262144;
    constexpr int64_t L1_SIZE = 1048576;
    // 2x3  2x3 
    auto num_axes = input.dim();

    if(dim!=-1 && dim < num_axes && (dim!= (num_axes - 1)) ) {
         auto input_reorder = input.transpose(dim,num_axes - 1).contiguous(); 
    }
    int device_id;
    aclrtGetDevice(&device_id);
    auto npu_stream = c10_npu::getCurrentNPUStream(device_id);
    auto acl_stream = npu_stream.stream();
    auto output = at::empty_like(index, input.options());
    
    auto axis_1 = index.sizes()[0];
   
    logme("Axis " << axis_1);
   
    uint32_t blockDim = 2;
    uint32_t totalLength = 1;

    for (uint32_t size : input.sizes()) {
        totalLength *= size;
    }

    logme("Axis " << axis_1  << " Total Length = " << totalLength);

    uint8_t* input_ptr = reinterpret_cast<uint8_t*>(input.storage().data_ptr().get());
    uint8_t* index_ptr = reinterpret_cast<uint8_t*>(index.storage().data_ptr().get());
    uint8_t* output_ptr = reinterpret_cast<uint8_t*>(output.storage().data_ptr().get());

    launch_gather(
        blockDim,
        // nullptr,
        acl_stream,
        input_ptr,
        dim,
        index_ptr,
        output_ptr,
        totalLength
    );
    aclrtSynchronizeStream(acl_stream);
    return output;
}
} // namespace my_add

PYBIND11_MODULE(_C, m)
{
    m.doc() = "gather_custom pybind11 interfaces";
    m.def("run_gather_custom", &my_add::run_gather_custom, "");
}