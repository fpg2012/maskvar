// JIT-loaded CPU helpers for interaction sampling.

#include <torch/extension.h>

#include <ATen/Parallel.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

namespace py = pybind11;

struct BoundingBox {
    int left;
    int right;
    int top;
    int bottom;
    bool has_foreground;
};

template <typename scalar_t>
bool row_has_any(const scalar_t* row_ptr, int width) {
    for (int x = 0; x < width; ++x) {
        if (static_cast<bool>(row_ptr[x])) {
            return true;
        }
    }
    return false;
}

void edt_1d(const float* source, float* output, int size) {
    const float inf = std::numeric_limits<float>::infinity();
    std::vector<int> vertices(size);
    std::vector<float> intersections(size + 1);
    int top = 0;
    vertices[0] = 0;
    intersections[0] = -inf;
    intersections[1] = inf;

    for (int idx = 1; idx < size; ++idx) {
        float split = 0.0f;
        while (true) {
            const int prev = vertices[top];
            const float numerator =
                (source[idx] + static_cast<float>(idx * idx)) -
                (source[prev] + static_cast<float>(prev * prev));
            const float denominator = static_cast<float>(2 * (idx - prev));
            split = numerator / denominator;
            if (split <= intersections[top]) {
                --top;
                if (top < 0) {
                    top = 0;
                    break;
                }
            } else {
                break;
            }
        }
        ++top;
        vertices[top] = idx;
        intersections[top] = split;
        intersections[top + 1] = inf;
    }

    top = 0;
    for (int idx = 0; idx < size; ++idx) {
        while (intersections[top + 1] < static_cast<float>(idx)) {
            ++top;
        }
        const float diff = static_cast<float>(idx - vertices[top]);
        output[idx] = diff * diff + source[vertices[top]];
    }
}

template <typename scalar_t>
void scan_row_edges(const scalar_t* row_ptr, int width, int* left, int* right) {
    *left = width;
    *right = -1;
    int lo = 0;
    int hi = width - 1;

    while (lo <= hi && (*left == width || *right == -1)) {
        if (*left == width && static_cast<bool>(row_ptr[lo])) {
            *left = lo;
        }
        if (*right == -1 && static_cast<bool>(row_ptr[hi])) {
            *right = hi;
        }
        ++lo;
        --hi;
    }

    if (*left == width) {
        *left = -1;
    }
    if (*right == -1) {
        *right = *left;
    }
}

template <typename scalar_t>
BoundingBox scan_bbox(const scalar_t* mask_ptr, int height, int width) {
    BoundingBox box{width, -1, height, -1, false};
    for (int y = 0; y < height; ++y) {
        const auto* row_ptr = mask_ptr + static_cast<int64_t>(y) * width;
        for (int x = 0; x < width; ++x) {
            if (!static_cast<bool>(row_ptr[x])) {
                continue;
            }
            box.has_foreground = true;
            box.left = std::min(box.left, x);
            box.right = std::max(box.right, x);
            box.top = std::min(box.top, y);
            box.bottom = std::max(box.bottom, y);
        }
    }

    return box;
}

template <typename scalar_t>
void scan_row_edges_pair(
    const scalar_t* negative_ptr,
    const scalar_t* positive_ptr,
    bool scan_negative,
    bool scan_positive,
    int width,
    int* negative_left,
    int* negative_right,
    int* positive_left,
    int* positive_right) {
    *negative_left = -1;
    *negative_right = -1;
    *positive_left = -1;
    *positive_right = -1;

    const bool negative_any = scan_negative && row_has_any(negative_ptr, width);
    const bool positive_any = scan_positive && row_has_any(positive_ptr, width);
    if (!negative_any && !positive_any) {
        return;
    }

    int lo = 0;
    int hi = width - 1;
    while (lo <= hi) {
        const bool need_negative =
            negative_any && (*negative_left < 0 || *negative_right < 0);
        const bool need_positive =
            positive_any && (*positive_left < 0 || *positive_right < 0);
        if (!need_negative && !need_positive) {
            break;
        }

        if (need_negative) {
            if (*negative_left < 0 && static_cast<bool>(negative_ptr[lo])) {
                *negative_left = lo;
            }
            if (*negative_right < 0 && static_cast<bool>(negative_ptr[hi])) {
                *negative_right = hi;
            }
        }
        if (need_positive) {
            if (*positive_left < 0 && static_cast<bool>(positive_ptr[lo])) {
                *positive_left = lo;
            }
            if (*positive_right < 0 && static_cast<bool>(positive_ptr[hi])) {
                *positive_right = hi;
            }
        }
        ++lo;
        --hi;
    }

    if (negative_any && *negative_right < 0) {
        *negative_right = *negative_left;
    }
    if (positive_any && *positive_right < 0) {
        *positive_right = *positive_left;
    }
}

template <typename scalar_t>
std::array<BoundingBox, 2> scan_pair_bbox(
    const scalar_t* negative_ptr,
    const scalar_t* positive_ptr,
    int height,
    int width) {
    BoundingBox negative_box{width, -1, height, -1, false};
    BoundingBox positive_box{width, -1, height, -1, false};
    for (int y = 0; y < height; ++y) {
        const auto* negative_row = negative_ptr + static_cast<int64_t>(y) * width;
        const auto* positive_row = positive_ptr + static_cast<int64_t>(y) * width;
        for (int x = 0; x < width; ++x) {
            if (static_cast<bool>(negative_row[x])) {
                negative_box.has_foreground = true;
                negative_box.left = std::min(negative_box.left, x);
                negative_box.right = std::max(negative_box.right, x);
                negative_box.top = std::min(negative_box.top, y);
                negative_box.bottom = std::max(negative_box.bottom, y);
            }
            if (static_cast<bool>(positive_row[x])) {
                positive_box.has_foreground = true;
                positive_box.left = std::min(positive_box.left, x);
                positive_box.right = std::max(positive_box.right, x);
                positive_box.top = std::min(positive_box.top, y);
                positive_box.bottom = std::max(positive_box.bottom, y);
            }
        }
    }

    return {negative_box, positive_box};
}

std::array<int64_t, 4> box_to_array(const BoundingBox& box) {
    if (!box.has_foreground) {
        return {0, 0, 0, 0};
    }
    return {
        static_cast<int64_t>(box.left),
        static_cast<int64_t>(box.top),
        static_cast<int64_t>(box.right + 1),
        static_cast<int64_t>(box.bottom + 1),
    };
}

template <typename scalar_t>
void compute_cropped_distance(
    const scalar_t* mask_ptr,
    const BoundingBox& bbox,
    int height,
    int width,
    float* output_ptr) {
    std::fill(output_ptr, output_ptr + static_cast<size_t>(height) * width, 0.0f);
    if (!bbox.has_foreground) {
        return;
    }

    const int left = bbox.left;
    const int top = bbox.top;
    const int right = bbox.right + 1;
    const int bottom = bbox.bottom + 1;
    const int crop_h = bottom - top;
    const int crop_w = right - left;
    const int padded_h = crop_h + 2;
    const int padded_w = crop_w + 2;
    const float inf = 1e20f;

    std::vector<float> source(static_cast<size_t>(padded_h) * padded_w, 0.0f);
    std::vector<float> tmp(static_cast<size_t>(padded_h) * padded_w, 0.0f);
    std::vector<float> dist(static_cast<size_t>(padded_h) * padded_w, 0.0f);

    for (int y = 0; y < crop_h; ++y) {
        const int src_row = (top + y) * width + left;
        const int dst_row = (y + 1) * padded_w + 1;
        for (int x = 0; x < crop_w; ++x) {
            source[dst_row + x] = static_cast<bool>(mask_ptr[src_row + x]) ? inf : 0.0f;
        }
    }

    at::parallel_for(0, padded_h, 1, [&](int64_t begin, int64_t end) {
        for (int64_t y = begin; y < end; ++y) {
            edt_1d(
                &source[static_cast<size_t>(y) * padded_w],
                &tmp[static_cast<size_t>(y) * padded_w],
                padded_w);
        }
    });

    at::parallel_for(0, padded_w, 1, [&](int64_t begin, int64_t end) {
        std::vector<float> col_in(padded_h, 0.0f);
        std::vector<float> col_out(padded_h, 0.0f);
        for (int64_t x = begin; x < end; ++x) {
            for (int y = 0; y < padded_h; ++y) {
                col_in[y] = tmp[static_cast<size_t>(y) * padded_w + x];
            }
            edt_1d(col_in.data(), col_out.data(), padded_h);
            for (int y = 0; y < padded_h; ++y) {
                dist[static_cast<size_t>(y) * padded_w + x] = col_out[y];
            }
        }
    });

    at::parallel_for(0, crop_h, 1, [&](int64_t begin, int64_t end) {
        for (int64_t y = begin; y < end; ++y) {
            const int dst_y = top + static_cast<int>(y);
            const size_t src_row = static_cast<size_t>(y + 1) * padded_w + 1;
            const size_t dst_row = static_cast<size_t>(dst_y) * width + left;
            for (int x = 0; x < crop_w; ++x) {
                output_ptr[dst_row + x] = std::sqrt(dist[src_row + x]);
            }
        }
    });
}

template <typename scalar_t>
void compute_single_distance(
    const scalar_t* mask_ptr,
    int height,
    int width,
    float* output_ptr) {
    const BoundingBox bbox = scan_bbox(mask_ptr, height, width);
    compute_cropped_distance(mask_ptr, bbox, height, width, output_ptr);
}

torch::Tensor distance(torch::Tensor mask, bool boundary_padding) {
    if (!boundary_padding) {
        throw std::runtime_error("boundary_padding must be True.");
    }
    if (!mask.device().is_cpu()) {
        throw std::runtime_error("mask must be on CPU.");
    }
    if (mask.dim() != 2 && mask.dim() != 3) {
        throw std::runtime_error("mask must be 2D or 3D.");
    }
    if (mask.scalar_type() != torch::kUInt8 && mask.scalar_type() != torch::kBool) {
        throw std::runtime_error("mask must be uint8 or bool.");
    }

    auto mask_contig = mask.contiguous();
    const auto height = static_cast<int>(mask_contig.size(-2));
    const auto width = static_cast<int>(mask_contig.size(-1));
    const int64_t batch = mask_contig.dim() == 2 ? 1 : mask_contig.size(0);

    auto flat = mask_contig.view({batch, height, width});
    auto output = torch::zeros(
        {batch, height, width},
        torch::TensorOptions().dtype(torch::kFloat32));
    auto* output_ptr = output.data_ptr<float>();
    const int64_t plane = static_cast<int64_t>(height) * width;

    if (mask_contig.scalar_type() == torch::kUInt8) {
        const auto* mask_ptr = flat.data_ptr<uint8_t>();
        at::parallel_for(0, batch, 1, [&](int64_t begin, int64_t end) {
            for (int64_t idx = begin; idx < end; ++idx) {
                compute_single_distance(
                    mask_ptr + idx * plane,
                    height,
                    width,
                    output_ptr + idx * plane);
            }
        });
    } else {
        const auto* mask_ptr = flat.data_ptr<bool>();
        at::parallel_for(0, batch, 1, [&](int64_t begin, int64_t end) {
            for (int64_t idx = begin; idx < end; ++idx) {
                compute_single_distance(
                    mask_ptr + idx * plane,
                    height,
                    width,
                    output_ptr + idx * plane);
            }
        });
    }

    if (mask.dim() == 2) {
        return output[0];
    }
    return output;
}

std::array<int64_t, 4> bbox(torch::Tensor mask) {
    if (!mask.device().is_cpu()) {
        throw std::runtime_error("mask must be on CPU.");
    }
    if (mask.dim() != 2) {
        throw std::runtime_error("mask must be 2D.");
    }
    if (mask.scalar_type() != torch::kUInt8 && mask.scalar_type() != torch::kBool) {
        throw std::runtime_error("mask must be uint8 or bool.");
    }

    auto mask_contig = mask.contiguous();
    const int height = static_cast<int>(mask_contig.size(0));
    const int width = static_cast<int>(mask_contig.size(1));

    if (mask_contig.scalar_type() == torch::kUInt8) {
        return box_to_array(scan_bbox(mask_contig.data_ptr<uint8_t>(), height, width));
    }
    return box_to_array(scan_bbox(mask_contig.data_ptr<bool>(), height, width));
}

py::tuple distance_pair(
    torch::Tensor negative_mask,
    torch::Tensor positive_mask,
    bool boundary_padding) {
    if (!boundary_padding) {
        throw std::runtime_error("boundary_padding must be True.");
    }
    if (negative_mask.sizes() != positive_mask.sizes()) {
        throw std::runtime_error("negative_mask and positive_mask must share shape.");
    }
    if (!negative_mask.device().is_cpu() || !positive_mask.device().is_cpu()) {
        throw std::runtime_error("negative_mask and positive_mask must be on CPU.");
    }
    if (negative_mask.dim() != 2 || positive_mask.dim() != 2) {
        throw std::runtime_error("negative_mask and positive_mask must be 2D.");
    }
    if (negative_mask.scalar_type() != positive_mask.scalar_type()) {
        throw std::runtime_error("negative_mask and positive_mask must share dtype.");
    }
    if (
        negative_mask.scalar_type() != torch::kUInt8 &&
        negative_mask.scalar_type() != torch::kBool) {
        throw std::runtime_error("negative_mask and positive_mask must be uint8 or bool.");
    }

    auto negative_contig = negative_mask.contiguous();
    auto positive_contig = positive_mask.contiguous();
    const int height = static_cast<int>(negative_contig.size(0));
    const int width = static_cast<int>(negative_contig.size(1));
    auto negative_out = torch::zeros(
        {height, width},
        torch::TensorOptions().dtype(torch::kFloat32));
    auto positive_out = torch::zeros_like(negative_out);

    if (negative_contig.scalar_type() == torch::kUInt8) {
        const auto* negative_ptr = negative_contig.data_ptr<uint8_t>();
        const auto* positive_ptr = positive_contig.data_ptr<uint8_t>();
        const auto boxes = scan_pair_bbox(negative_ptr, positive_ptr, height, width);
        compute_cropped_distance(
            negative_ptr,
            boxes[0],
            height,
            width,
            negative_out.data_ptr<float>());
        compute_cropped_distance(
            positive_ptr,
            boxes[1],
            height,
            width,
            positive_out.data_ptr<float>());
    } else {
        const auto* negative_ptr = negative_contig.data_ptr<bool>();
        const auto* positive_ptr = positive_contig.data_ptr<bool>();
        const auto boxes = scan_pair_bbox(negative_ptr, positive_ptr, height, width);
        compute_cropped_distance(
            negative_ptr,
            boxes[0],
            height,
            width,
            negative_out.data_ptr<float>());
        compute_cropped_distance(
            positive_ptr,
            boxes[1],
            height,
            width,
            positive_out.data_ptr<float>());
    }

    return py::make_tuple(negative_out, positive_out);
}

py::tuple bbox_pair(torch::Tensor negative_mask, torch::Tensor positive_mask) {
    if (negative_mask.sizes() != positive_mask.sizes()) {
        throw std::runtime_error("negative_mask and positive_mask must share shape.");
    }
    if (!negative_mask.device().is_cpu() || !positive_mask.device().is_cpu()) {
        throw std::runtime_error("negative_mask and positive_mask must be on CPU.");
    }
    if (negative_mask.dim() != 2 || positive_mask.dim() != 2) {
        throw std::runtime_error("negative_mask and positive_mask must be 2D.");
    }
    if (negative_mask.scalar_type() != positive_mask.scalar_type()) {
        throw std::runtime_error("negative_mask and positive_mask must share dtype.");
    }
    if (
        negative_mask.scalar_type() != torch::kUInt8 &&
        negative_mask.scalar_type() != torch::kBool) {
        throw std::runtime_error("negative_mask and positive_mask must be uint8 or bool.");
    }

    auto negative_contig = negative_mask.contiguous();
    auto positive_contig = positive_mask.contiguous();
    const int height = static_cast<int>(negative_contig.size(0));
    const int width = static_cast<int>(negative_contig.size(1));

    if (negative_contig.scalar_type() == torch::kUInt8) {
        const auto boxes = scan_pair_bbox(
            negative_contig.data_ptr<uint8_t>(),
            positive_contig.data_ptr<uint8_t>(),
            height,
            width);
        return py::make_tuple(box_to_array(boxes[0]), box_to_array(boxes[1]));
    }
    const auto boxes = scan_pair_bbox(
        negative_contig.data_ptr<bool>(),
        positive_contig.data_ptr<bool>(),
        height,
        width);
    return py::make_tuple(box_to_array(boxes[0]), box_to_array(boxes[1]));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("distance", &distance, "CPU distance transform");
    module.def("bbox", &bbox, "CPU bbox scan");
    module.def("distance_pair", &distance_pair, "CPU pair distance transform");
    module.def("bbox_pair", &bbox_pair, "CPU pair bbox scan");
}
