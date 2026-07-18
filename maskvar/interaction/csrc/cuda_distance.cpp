#include <torch/extension.h>

#include <vector>

torch::Tensor cuda_distance(torch::Tensor mask);
torch::Tensor cuda_voronoi(torch::Tensor mask);
std::vector<torch::Tensor> cuda_distance_pair(
    torch::Tensor negative_mask,
    torch::Tensor positive_mask
);
std::vector<torch::Tensor> cuda_bbox_pair(
    torch::Tensor negative_mask,
    torch::Tensor positive_mask
);
std::vector<torch::Tensor> cuda_sample_from_error_pair(
    torch::Tensor negative_mask,
    torch::Tensor positive_mask,
    double sfc_inner_k,
    int64_t random_value,
    bool need_debug
);
std::vector<torch::Tensor> cuda_sample_from_masks(
    torch::Tensor pred_mask,
    torch::Tensor gt_mask,
    torch::Tensor ignore_mask,
    double sfc_inner_k,
    int64_t random_value,
    bool need_debug
);
std::vector<torch::Tensor> cuda_sample_from_masks_batch(
    torch::Tensor pred_mask,
    torch::Tensor gt_mask,
    torch::Tensor ignore_mask,
    double sfc_inner_k,
    torch::Tensor random_values
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("distance", &cuda_distance, "CUDA exact distance");
    module.def("voronoi", &cuda_voronoi, "CUDA Voronoi");
    module.def("distance_pair", &cuda_distance_pair, "CUDA exact distance pair");
    module.def("bbox_pair", &cuda_bbox_pair, "CUDA bbox pair");
    module.def(
        "sample_from_error_pair",
        &cuda_sample_from_error_pair,
        "CUDA click sampling from error pair"
    );
    module.def(
        "sample_from_masks",
        &cuda_sample_from_masks,
        "CUDA click sampling from masks"
    );
    module.def(
        "sample_from_masks_batch",
        &cuda_sample_from_masks_batch,
        "CUDA batched click sampling from masks"
    );
}
