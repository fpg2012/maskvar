#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace {

constexpr int BLOCKSIZE = 64;
constexpr short MARKER = -32768;

#define TOID(x, y, size) (__mul24((y), (size)) + (x))

__global__ void kernelFloodDown(short2 *input, short2 *output, int size, int band_size) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * band_size;
    int id = TOID(tx, ty, size);
    short2 pixel1 = make_short2(MARKER, MARKER);
    short2 pixel2;

    for (int i = 0; i < band_size; ++i, id += size) {
        pixel2 = input[id];
        if (pixel2.x != MARKER) {
            pixel1 = pixel2;
        }
        output[id] = pixel1;
    }
}

__global__ void kernelFloodUp(short2 *input, short2 *output, int size, int band_size) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = (blockIdx.y + 1) * band_size - 1;
    int id = TOID(tx, ty, size);
    short2 pixel1 = make_short2(MARKER, MARKER);
    short2 pixel2;
    int dist1;
    int dist2;

    for (int i = 0; i < band_size; ++i, id -= size) {
        dist1 = abs(pixel1.y - ty + i);
        pixel2 = input[id];
        dist2 = abs(pixel2.y - ty + i);
        if (dist2 < dist1) {
            pixel1 = pixel2;
        }
        output[id] = pixel1;
    }
}

__global__ void kernelPropagateInterband(
    short2 *input,
    short2 *output,
    int size,
    int band_size
) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int inc = band_size * size;
    int ny;
    int nid;
    int n_dist;
    short2 pixel;

    int ty = blockIdx.y * band_size;
    int top_id = TOID(tx, ty, size);
    int bottom_id = TOID(tx, ty + band_size - 1, size);
    int tid = blockIdx.y * size + tx;
    int bid = tid + (size * size / band_size);

    pixel = input[top_id];
    int my_dist = abs(pixel.y - ty);
    output[tid] = pixel;

    for (nid = bottom_id - inc; nid >= 0; nid -= inc) {
        pixel = input[nid];
        if (pixel.x != MARKER) {
            n_dist = abs(pixel.y - ty);
            if (n_dist < my_dist) {
                output[tid] = pixel;
            }
            break;
        }
    }

    ty = ty + band_size - 1;
    pixel = input[bottom_id];
    my_dist = abs(pixel.y - ty);
    output[bid] = pixel;

    for (ny = ty + 1, nid = top_id + inc; ny < size; ny += band_size, nid += inc) {
        pixel = input[nid];
        if (pixel.x != MARKER) {
            n_dist = abs(pixel.y - ty);
            if (n_dist < my_dist) {
                output[bid] = pixel;
            }
            break;
        }
    }
}

__global__ void kernelUpdateVertical(
    short2 *color,
    short2 *margin,
    short2 *output,
    int size,
    int band_size
) {
    __shared__ short2 block[BLOCKSIZE][BLOCKSIZE];

    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * band_size;

    short2 top = margin[blockIdx.y * size + tx];
    short2 bottom = margin[(blockIdx.y + size / band_size) * size + tx];
    short2 pixel;

    int dist;
    int my_dist;
    int id = TOID(tx, ty, size);
    int n_step = band_size / blockDim.x;

    for (int step = 0; step < n_step; ++step) {
        int y_start = blockIdx.y * band_size + step * blockDim.x;
        int y_end = y_start + blockDim.x;

        for (ty = y_start; ty < y_end; ++ty, id += size) {
            pixel = color[id];
            my_dist = abs(pixel.y - ty);

            dist = abs(top.y - ty);
            if (dist < my_dist) {
                my_dist = dist;
                pixel = top;
            }

            dist = abs(bottom.y - ty);
            if (dist < my_dist) {
                pixel = bottom;
            }

            block[threadIdx.x][ty - y_start] = make_short2(pixel.y, pixel.x);
        }

        __syncthreads();

        int tid = TOID(
            blockIdx.y * band_size + step * blockDim.x + threadIdx.x,
            blockIdx.x * blockDim.x,
            size
        );

        for (int i = 0; i < blockDim.x; ++i, tid += size) {
            output[tid] = block[i][threadIdx.x];
        }

        __syncthreads();
    }
}

__device__ bool dominate(
    long long x1,
    long long y1,
    long long x2,
    long long y2,
    long long x3,
    long long y3,
    long long x0
) {
    long long k1 = y2 - y1;
    long long k2 = y3 - y2;
    return (
        (k1 * (y1 + y2) + (x2 - x1) * ((x1 + x2) - (x0 << 1))) * k2 >
        (k2 * (y2 + y3) + (x3 - x2) * ((x2 + x3) - (x0 << 1))) * k1
    );
}

__global__ void kernelProximatePoints(short2 *input, short2 *stack, int size, int band_size) {
    int tx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    int ty = __mul24(blockIdx.y, band_size);
    int id = TOID(tx, ty, size);
    int lasty = -1;
    short2 last1;
    short2 last2;
    short2 current;

    last1.y = -1;
    last2.y = -1;

    for (int i = 0; i < band_size; ++i, id += size) {
        current = input[id];
        if (current.x != MARKER) {
            while (last2.y >= 0) {
                if (!dominate(last1.x, last2.y, last2.x, lasty, current.x, current.y, tx)) {
                    break;
                }

                lasty = last2.y;
                last2 = last1;

                if (last1.y >= 0) {
                    last1 = stack[TOID(tx, last1.y, size)];
                }
            }

            last1 = last2;
            last2 = make_short2(current.x, lasty);
            lasty = current.y;
            stack[id] = last2;
        }
    }

    if (lasty != ty + band_size - 1) {
        stack[TOID(tx, ty + band_size - 1, size)] = make_short2(MARKER, lasty);
    }
}

__global__ void kernelCreateForwardPointers(
    short2 *input,
    short2 *output,
    int size,
    int band_size
) {
    int tx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    int ty = __mul24(blockIdx.y + 1, band_size) - 1;
    int id = TOID(tx, ty, size);
    int lasty = -1;
    int nexty;
    short2 current;

    current = input[id];
    if (current.x == MARKER) {
        nexty = current.y;
    } else {
        nexty = ty;
    }

    for (int i = 0; i < band_size; ++i, id -= size) {
        if (ty - i == nexty) {
            current = make_short2(lasty, input[id].y);
            output[id] = current;
            lasty = nexty;
            nexty = current.y;
        }
    }

    if (lasty != ty - band_size + 1) {
        output[id + size] = make_short2(lasty, MARKER);
    }
}

__global__ void kernelMergeBands(short2 *color, short2 *link, short2 *output, int size, int band_size) {
    int tx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    int band1 = blockIdx.y * 2;
    int band2 = band1 + 1;
    int firsty;
    int lasty;
    short2 last1;
    short2 last2;
    short2 current;

    lasty = __mul24(band2, band_size) - 1;
    last2 = make_short2(
        color[TOID(tx, lasty, size)].x,
        link[TOID(tx, lasty, size)].y
    );

    if (last2.x == MARKER) {
        lasty = last2.y;
        if (lasty >= 0) {
            last2 = make_short2(
                color[TOID(tx, lasty, size)].x,
                link[TOID(tx, lasty, size)].y
            );
        } else {
            last2 = make_short2(MARKER, MARKER);
        }
    }

    if (last2.y >= 0) {
        last1 = make_short2(
            color[TOID(tx, last2.y, size)].x,
            link[TOID(tx, last2.y, size)].y
        );
    }

    firsty = __mul24(band2, band_size);
    current = make_short2(
        link[TOID(tx, firsty, size)].x,
        color[TOID(tx, firsty, size)].x
    );

    if (current.y == MARKER) {
        firsty = current.x;
        if (firsty >= 0) {
            current = make_short2(
                link[TOID(tx, firsty, size)].x,
                color[TOID(tx, firsty, size)].x
            );
        } else {
            current = make_short2(MARKER, MARKER);
        }
    }

    int top = 0;

    while (top < 2 && current.y >= 0) {
        while (last2.y >= 0) {
            if (!dominate(last1.x, last2.y, last2.x, lasty, current.y, firsty, tx)) {
                break;
            }

            lasty = last2.y;
            last2 = last1;
            top--;

            if (last1.y >= 0) {
                last1 = make_short2(
                    color[TOID(tx, last1.y, size)].x,
                    link[TOID(tx, last1.y, size)].y
                );
            }
        }

        output[TOID(tx, firsty, size)] = make_short2(current.x, lasty);

        if (lasty >= 0) {
            output[TOID(tx, lasty, size)] = make_short2(firsty, last2.y);
        }

        last1 = last2;
        last2 = make_short2(current.y, lasty);
        lasty = firsty;
        firsty = current.x;
        top = max(1, top + 1);

        if (firsty >= 0) {
            current = make_short2(
                link[TOID(tx, firsty, size)].x,
                color[TOID(tx, firsty, size)].x
            );
        } else {
            current = make_short2(MARKER, MARKER);
        }
    }

    firsty = __mul24(band1, band_size);
    lasty = __mul24(band2, band_size);
    current = link[TOID(tx, firsty, size)];

    if (current.y == MARKER && current.x < 0) {
        last1 = link[TOID(tx, lasty, size)];
        if (last1.y == MARKER) {
            current.x = last1.x;
        } else {
            current.x = lasty;
        }
        output[TOID(tx, firsty, size)] = current;
    }

    firsty = __mul24(band1, band_size) + band_size - 1;
    lasty = __mul24(band2, band_size) + band_size - 1;
    current = link[TOID(tx, lasty, size)];

    if (current.x == MARKER && current.y < 0) {
        last1 = link[TOID(tx, firsty, size)];
        if (last1.x == MARKER) {
            current.y = last1.y;
        } else {
            current.y = firsty;
        }
        output[TOID(tx, lasty, size)] = current;
    }
}

__global__ void kernelDoubleToSingleList(short2 *color, short2 *link, short2 *output, int size) {
    int tx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    int ty = blockIdx.y;
    int id = TOID(tx, ty, size);
    output[id] = make_short2(color[id].x, link[id].y);
}

__global__ void kernelColor(short2 *input, short2 *output, int size) {
    __shared__ short2 block[BLOCKSIZE][BLOCKSIZE];

    int col = threadIdx.x;
    int tid = threadIdx.y;
    int tx = __mul24(blockIdx.x, blockDim.x) + col;
    int dx;
    int dy;
    int lasty;
    unsigned int best;
    unsigned int dist;
    short2 last1;
    short2 last2;

    lasty = size - 1;
    last2 = input[TOID(tx, lasty, size)];

    if (last2.x == MARKER) {
        lasty = last2.y;
        last2 = input[TOID(tx, lasty, size)];
    }

    if (last2.y >= 0) {
        last1 = input[TOID(tx, last2.y, size)];
    }

    int n_step = size / blockDim.x;
    for (int step = 0; step < n_step; ++step) {
        int y_start = size - step * blockDim.x - 1;
        int y_end = size - (step + 1) * blockDim.x;

        for (int ty = y_start - tid; ty >= y_end; ty -= blockDim.y) {
            dx = last2.x - tx;
            dy = lasty - ty;
            best = dist = __mul24(dx, dx) + __mul24(dy, dy);

            while (last2.y >= 0) {
                dx = last1.x - tx;
                dy = last2.y - ty;
                dist = __mul24(dx, dx) + __mul24(dy, dy);
                if (dist > best) {
                    break;
                }

                best = dist;
                lasty = last2.y;
                last2 = last1;

                if (last2.y >= 0) {
                    last1 = input[TOID(tx, last2.y, size)];
                }
            }

            block[threadIdx.x][ty - y_end] = make_short2(lasty, last2.x);
        }

        __syncthreads();

        if (!threadIdx.y) {
            int id = TOID(y_end + threadIdx.x, blockIdx.x * blockDim.x, size);
            for (int i = 0; i < blockDim.x; ++i, id += size) {
                output[id] = block[i][threadIdx.x];
            }
        }

        __syncthreads();
    }
}

__global__ void build_sites_kernel(
    const bool *mask,
    short2 *sites,
    int height,
    int width,
    int texture_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= texture_size || y >= texture_size) {
        return;
    }

    short2 value = make_short2(static_cast<short>(x), static_cast<short>(y));
    if (y >= 1 && y <= height && x >= 1 && x <= width) {
        if (mask[(y - 1) * width + (x - 1)]) {
            value = make_short2(MARKER, MARKER);
        }
    }
    sites[TOID(x, y, texture_size)] = value;
}

__global__ void coords_to_distance_kernel(
    const short2 *coords,
    float *distance,
    int height,
    int width,
    int texture_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    int tx = x + 1;
    int ty = y + 1;
    short2 site = coords[TOID(tx, ty, texture_size)];
    if (site.x == MARKER) {
        distance[y * width + x] = 0.0f;
        return;
    }

    int dx = static_cast<int>(site.x) - tx;
    int dy = static_cast<int>(site.y) - ty;
    distance[y * width + x] = sqrtf(static_cast<float>(dx * dx + dy * dy));
}

__global__ void build_error_masks_kernel(
    const bool *pred,
    const bool *gt,
    const bool *ignore,
    bool *false_negative,
    bool *false_positive,
    int numel
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numel) {
        return;
    }

    bool pred_value = pred[index];
    bool gt_value = gt[index];
    bool valid = !ignore[index];
    false_negative[index] = (!pred_value && gt_value) && valid;
    false_positive[index] = (pred_value && !gt_value) && valid;
}

__global__ void reduce_bbox_pair_kernel(
    const bool *negative_mask,
    const bool *positive_mask,
    int height,
    int width,
    int *negative_bbox,
    int *positive_bbox
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int numel = height * width;
    if (index >= numel) {
        return;
    }

    int y = index / width;
    int x = index % width;
    if (negative_mask[index]) {
        atomicMin(&negative_bbox[0], x);
        atomicMin(&negative_bbox[1], y);
        atomicMax(&negative_bbox[2], x + 1);
        atomicMax(&negative_bbox[3], y + 1);
    }
    if (positive_mask[index]) {
        atomicMin(&positive_bbox[0], x);
        atomicMin(&positive_bbox[1], y);
        atomicMax(&positive_bbox[2], x + 1);
        atomicMax(&positive_bbox[3], y + 1);
    }
}

__global__ void fill_crop_from_bbox_kernel(
    const bool *src,
    bool *dst,
    int src_width,
    int left,
    int up,
    int crop_height,
    int crop_width
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= crop_width || y >= crop_height) {
        return;
    }

    dst[y * crop_width + x] = src[(up + y) * src_width + left + x];
}

__global__ void write_crop_to_full_kernel(
    const float *crop,
    float *full,
    int full_width,
    int left,
    int up,
    int crop_height,
    int crop_width
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= crop_width || y >= crop_height) {
        return;
    }

    full[(up + y) * full_width + left + x] = crop[y * crop_width + x];
}

int next_power_of_two(int value) {
    int result = 64;
    while (result < value) {
        result <<= 1;
    }
    return result;
}

int choose_phase1_band(int texture_size) {
    return std::min(32, texture_size / 64);
}

std::array<int, 4> bbox_to_host(torch::Tensor bbox) {
    auto host = bbox.to(torch::kCPU);
    const int *values = host.data_ptr<int>();
    return {values[0], values[1], values[2], values[3]};
}

bool bbox_is_empty_host(const int *bbox) {
    return bbox[0] == std::numeric_limits<int>::max() ||
        bbox[1] == std::numeric_limits<int>::max() ||
        bbox[2] <= bbox[0] ||
        bbox[3] <= bbox[1];
}

torch::Tensor build_bbox_tensor(torch::Device device) {
    auto bbox = torch::empty({4}, torch::TensorOptions().device(device).dtype(torch::kInt32));
    auto init = torch::tensor(
        {
            std::numeric_limits<int>::max(),
            std::numeric_limits<int>::max(),
            0,
            0,
        },
        torch::TensorOptions().dtype(torch::kInt32)
    );
    bbox.copy_(init.to(device));
    return bbox;
}

std::vector<torch::Tensor> compute_bbox_pair(
    torch::Tensor negative_mask,
    torch::Tensor positive_mask
) {
    int64_t height = negative_mask.size(0);
    int64_t width = negative_mask.size(1);
    auto negative_bbox = build_bbox_tensor(negative_mask.device());
    auto positive_bbox = build_bbox_tensor(negative_mask.device());
    int numel = static_cast<int>(height * width);
    int block = 256;
    int grid = (numel + block - 1) / block;
    auto stream = at::cuda::getDefaultCUDAStream(negative_mask.get_device());
    reduce_bbox_pair_kernel<<<grid, block, 0, stream>>>(
        negative_mask.data_ptr<bool>(),
        positive_mask.data_ptr<bool>(),
        static_cast<int>(height),
        static_cast<int>(width),
        negative_bbox.data_ptr<int>(),
        positive_bbox.data_ptr<int>()
    );
    return {negative_bbox, positive_bbox};
}

torch::Tensor normalize_bbox_tensor(
    torch::Tensor bbox,
    const std::array<int, 4> &bbox_host
) {
    if (!bbox_is_empty_host(bbox_host.data())) {
        return bbox;
    }
    return torch::zeros_like(bbox);
}

torch::Tensor crop_from_bbox_host(torch::Tensor mask, const std::array<int, 4> &bbox) {
    int left = bbox[0];
    int up = bbox[1];
    int right = bbox[2];
    int bottom = bbox[3];
    int crop_height = bottom - up;
    int crop_width = right - left;

    auto crop = torch::empty(
        {crop_height, crop_width},
        torch::TensorOptions().device(mask.device()).dtype(torch::kBool)
    );
    dim3 block(16, 16);
    dim3 grid(
        (crop_width + block.x - 1) / block.x,
        (crop_height + block.y - 1) / block.y
    );
    auto stream = at::cuda::getDefaultCUDAStream(mask.get_device());
    fill_crop_from_bbox_kernel<<<grid, block, 0, stream>>>(
        mask.data_ptr<bool>(),
        crop.data_ptr<bool>(),
        static_cast<int>(mask.size(1)),
        left,
        up,
        crop_height,
        crop_width
    );
    return crop;
}

torch::Tensor build_distance_full_host(
    torch::Tensor crop_dist,
    const std::array<int, 4> &bbox,
    int64_t height,
    int64_t width
) {
    auto full = torch::zeros(
        {height, width},
        torch::TensorOptions().device(crop_dist.device()).dtype(torch::kFloat32)
    );
    int left = bbox[0];
    int up = bbox[1];
    int crop_height = static_cast<int>(crop_dist.size(0));
    int crop_width = static_cast<int>(crop_dist.size(1));
    dim3 block(16, 16);
    dim3 grid(
        (crop_width + block.x - 1) / block.x,
        (crop_height + block.y - 1) / block.y
    );
    auto stream = at::cuda::getDefaultCUDAStream(crop_dist.get_device());
    write_crop_to_full_kernel<<<grid, block, 0, stream>>>(
        crop_dist.data_ptr<float>(),
        full.data_ptr<float>(),
        static_cast<int>(width),
        left,
        up,
        crop_height,
        crop_width
    );
    return full;
}

float compute_distance_scale(double sfc_inner_k) {
    if (sfc_inner_k < 0.0) {
        return 0.0f;
    }
    TORCH_CHECK(sfc_inner_k >= 1.0, "sfc_inner_k must be >= 1.0 or < 0.0.");
    return static_cast<float>(1.0 / (sfc_inner_k + std::numeric_limits<float>::epsilon()));
}

void run_pba_2d(short2 *tex0, short2 *tex1, short2 *margin, int texture_size, cudaStream_t stream) {
    int m1 = choose_phase1_band(texture_size);
    int m2 = std::min(32, texture_size);
    int m3 = 2;

    dim3 block_1d(BLOCKSIZE);
    dim3 grid_phase1(texture_size / block_1d.x, m1);
    int band1 = texture_size / m1;
    kernelFloodDown<<<grid_phase1, block_1d, 0, stream>>>(tex0, tex0, texture_size, band1);
    kernelFloodUp<<<grid_phase1, block_1d, 0, stream>>>(tex0, tex0, texture_size, band1);
    kernelPropagateInterband<<<grid_phase1, block_1d, 0, stream>>>(tex0, margin, texture_size, band1);
    kernelUpdateVertical<<<grid_phase1, block_1d, 0, stream>>>(tex0, margin, tex1, texture_size, band1);

    dim3 grid_phase2(texture_size / block_1d.x, m2);
    int band2 = texture_size / m2;
    kernelProximatePoints<<<grid_phase2, block_1d, 0, stream>>>(tex1, tex0, texture_size, band2);
    kernelCreateForwardPointers<<<grid_phase2, block_1d, 0, stream>>>(tex0, tex0, texture_size, band2);

    for (int no_band = m2; no_band > 1; no_band /= 2) {
        dim3 grid_merge(texture_size / block_1d.x, no_band / 2);
        kernelMergeBands<<<grid_merge, block_1d, 0, stream>>>(
            tex1,
            tex0,
            tex0,
            texture_size,
            texture_size / no_band
        );
    }

    dim3 grid_single(texture_size / block_1d.x, texture_size);
    kernelDoubleToSingleList<<<grid_single, block_1d, 0, stream>>>(tex1, tex0, tex0, texture_size);

    dim3 block_2d(BLOCKSIZE, m3);
    dim3 grid_phase3(texture_size / block_2d.x);
    kernelColor<<<grid_phase3, block_2d, 0, stream>>>(tex0, tex1, texture_size);
}

torch::Tensor distance_impl(torch::Tensor mask) {
    TORCH_CHECK(mask.is_cuda(), "mask must be a CUDA tensor.");
    TORCH_CHECK(mask.dim() == 2, "mask must be 2D.");

    auto guard = c10::cuda::CUDAGuard(mask.device());
    auto mask_bool = mask.to(torch::kBool).contiguous();

    int64_t height = mask_bool.size(0);
    int64_t width = mask_bool.size(1);
    int texture_size = next_power_of_two(static_cast<int>(std::max(height + 2, width + 2)));
    int phase1_band = choose_phase1_band(texture_size);

    auto short_options = torch::TensorOptions().device(mask.device()).dtype(torch::kInt16);
    auto float_options = torch::TensorOptions().device(mask.device()).dtype(torch::kFloat32);

    auto tex0 = torch::empty({texture_size, texture_size, 2}, short_options);
    auto tex1 = torch::empty({texture_size, texture_size, 2}, short_options);
    auto margin = torch::empty({phase1_band * 2, texture_size, 2}, short_options);
    auto dist = torch::empty({height, width}, float_options);

    dim3 block(16, 16);
    dim3 grid(
        (texture_size + block.x - 1) / block.x,
        (texture_size + block.y - 1) / block.y
    );
    auto stream = at::cuda::getDefaultCUDAStream(mask.get_device());

    build_sites_kernel<<<grid, block, 0, stream>>>(
        mask_bool.data_ptr<bool>(),
        reinterpret_cast<short2 *>(tex0.data_ptr<int16_t>()),
        static_cast<int>(height),
        static_cast<int>(width),
        texture_size
    );

    run_pba_2d(
        reinterpret_cast<short2 *>(tex0.data_ptr<int16_t>()),
        reinterpret_cast<short2 *>(tex1.data_ptr<int16_t>()),
        reinterpret_cast<short2 *>(margin.data_ptr<int16_t>()),
        texture_size,
        stream
    );

    dim3 out_grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );
    coords_to_distance_kernel<<<out_grid, block, 0, stream>>>(
        reinterpret_cast<short2 *>(tex1.data_ptr<int16_t>()),
        dist.data_ptr<float>(),
        static_cast<int>(height),
        static_cast<int>(width),
        texture_size
    );

    return dist;
}

std::vector<torch::Tensor> sample_from_error_pair_impl(
    torch::Tensor negative_mask,
    torch::Tensor positive_mask,
    double sfc_inner_k,
    int64_t random_value,
    bool need_debug
) {
    TORCH_CHECK(negative_mask.is_cuda(), "negative_mask must be CUDA.");
    TORCH_CHECK(positive_mask.is_cuda(), "positive_mask must be CUDA.");
    TORCH_CHECK(negative_mask.dim() == 2, "negative_mask must be 2D.");
    TORCH_CHECK(positive_mask.dim() == 2, "positive_mask must be 2D.");
    TORCH_CHECK(
        negative_mask.sizes() == positive_mask.sizes(),
        "negative_mask and positive_mask must share the same shape."
    );

    auto guard = c10::cuda::CUDAGuard(negative_mask.device());
    auto negative_bool = negative_mask.to(torch::kBool).contiguous();
    auto positive_bool = positive_mask.to(torch::kBool).contiguous();
    int64_t height = negative_bool.size(0);
    int64_t width = negative_bool.size(1);
    float dist_scale = compute_distance_scale(sfc_inner_k);

    auto bbox_pair = compute_bbox_pair(negative_bool, positive_bool);
    auto negative_bbox = bbox_pair[0];
    auto positive_bbox = bbox_pair[1];
    auto negative_bbox_host = bbox_to_host(negative_bbox);
    auto positive_bbox_host = bbox_to_host(positive_bbox);
    bool negative_empty = bbox_is_empty_host(negative_bbox_host.data());
    bool positive_empty = bbox_is_empty_host(positive_bbox_host.data());

    auto zero_dist = torch::zeros(
        {1, 1},
        torch::TensorOptions().device(negative_bool.device()).dtype(torch::kFloat32)
    );
    auto negative_crop_dist = zero_dist;
    auto positive_crop_dist = zero_dist;
    if (!negative_empty) {
        negative_crop_dist = distance_impl(crop_from_bbox_host(negative_bool, negative_bbox_host));
    }
    if (!positive_empty) {
        positive_crop_dist = distance_impl(crop_from_bbox_host(positive_bool, positive_bbox_host));
    }

    float negative_max = 0.0f;
    float positive_max = 0.0f;
    if (!negative_empty) {
        auto tensor = negative_crop_dist.max().to(torch::kCPU);
        negative_max = tensor.data_ptr<float>()[0];
    }
    if (!positive_empty) {
        auto tensor = positive_crop_dist.max().to(torch::kCPU);
        positive_max = tensor.data_ptr<float>()[0];
    }

    auto result = torch::full(
        {3},
        -1,
        torch::TensorOptions().device(negative_bool.device()).dtype(torch::kInt32)
    );
    auto meta = torch::zeros(
        {4},
        torch::TensorOptions().device(negative_bool.device()).dtype(torch::kFloat32)
    );
    meta[0] = negative_max;
    meta[1] = positive_max;

    auto empty_f32 = torch::empty(
        {0, 0},
        torch::TensorOptions().device(negative_bool.device()).dtype(torch::kFloat32)
    );
    auto empty_u8 = torch::empty(
        {0, 0},
        torch::TensorOptions().device(negative_bool.device()).dtype(torch::kUInt8)
    );
    auto negative_full = empty_f32;
    auto positive_full = empty_f32;
    auto candidate_full = empty_u8;
    auto false_negative_out = empty_u8;
    auto false_positive_out = empty_u8;

    if (need_debug) {
        negative_full = torch::zeros(
            {height, width},
            torch::TensorOptions().device(negative_bool.device()).dtype(torch::kFloat32)
        );
        positive_full = torch::zeros(
            {height, width},
            torch::TensorOptions().device(negative_bool.device()).dtype(torch::kFloat32)
        );
        candidate_full = torch::zeros(
            {height, width},
            torch::TensorOptions().device(negative_bool.device()).dtype(torch::kUInt8)
        );
        false_negative_out = negative_bool.to(torch::kUInt8);
        false_positive_out = positive_bool.to(torch::kUInt8);
        if (!negative_empty) {
            negative_full = build_distance_full_host(
                negative_crop_dist,
                negative_bbox_host,
                height,
                width
            );
        }
        if (!positive_empty) {
            positive_full = build_distance_full_host(
                positive_crop_dist,
                positive_bbox_host,
                height,
                width
            );
        }
    }

    if (negative_max == 0.0f && positive_max == 0.0f) {
        return {
            result,
            meta,
            negative_full,
            positive_full,
            candidate_full,
            false_negative_out,
            false_positive_out,
        };
    }

    bool use_negative = negative_max > positive_max;
    meta[2] = use_negative ? 1.0f : 0.0f;
    float threshold = use_negative ? dist_scale * negative_max : dist_scale * positive_max;
    meta[3] = threshold;

    torch::Tensor target_crop_dist = use_negative ? negative_crop_dist : positive_crop_dist;
    const auto &target_bbox_host = use_negative ? negative_bbox_host : positive_bbox_host;
    int left = target_bbox_host[0];
    int up = target_bbox_host[1];
    int crop_width = static_cast<int>(target_crop_dist.size(1));
    int numel = static_cast<int>(target_crop_dist.numel());

    auto candidate_crop = target_crop_dist > threshold;
    auto candidate_flat = torch::nonzero(candidate_crop.reshape({numel})).reshape({-1});
    int64_t candidate_count = candidate_flat.numel();
    if (candidate_count > 0) {
        int64_t target_index = random_value % candidate_count;
        auto selected = candidate_flat[target_index].to(torch::kCPU);
        int flat_value = static_cast<int>(selected.data_ptr<int64_t>()[0]);
        result[0] = flat_value / crop_width + up;
        result[1] = flat_value % crop_width + left;
        result[2] = use_negative ? 1 : 0;
    }

    if (need_debug) {
        auto candidate_debug = build_distance_full_host(
            candidate_crop.to(torch::kUInt8).to(torch::kFloat32),
            target_bbox_host,
            height,
            width
        );
        candidate_full = candidate_debug.to(torch::kUInt8);
    }

    return {
        result,
        meta,
        negative_full,
        positive_full,
        candidate_full,
        false_negative_out,
        false_positive_out,
    };
}

}  // namespace

torch::Tensor cuda_distance(torch::Tensor mask) {
    return distance_impl(mask);
}

torch::Tensor cuda_voronoi(torch::Tensor mask) {
    TORCH_CHECK(mask.is_cuda(), "mask must be a CUDA tensor.");
    TORCH_CHECK(mask.dim() == 2, "mask must be 2D.");

    auto guard = c10::cuda::CUDAGuard(mask.device());
    auto mask_bool = mask.to(torch::kBool).contiguous();
    int64_t height = mask_bool.size(0);
    int64_t width = mask_bool.size(1);
    int texture_size = next_power_of_two(static_cast<int>(std::max(height + 2, width + 2)));
    int phase1_band = choose_phase1_band(texture_size);

    auto short_options = torch::TensorOptions().device(mask.device()).dtype(torch::kInt16);
    auto tex0 = torch::empty({texture_size, texture_size, 2}, short_options);
    auto tex1 = torch::empty({texture_size, texture_size, 2}, short_options);
    auto margin = torch::empty({phase1_band * 2, texture_size, 2}, short_options);

    dim3 block(16, 16);
    dim3 grid(
        (texture_size + block.x - 1) / block.x,
        (texture_size + block.y - 1) / block.y
    );
    auto stream = at::cuda::getDefaultCUDAStream(mask.get_device());
    build_sites_kernel<<<grid, block, 0, stream>>>(
        mask_bool.data_ptr<bool>(),
        reinterpret_cast<short2 *>(tex0.data_ptr<int16_t>()),
        static_cast<int>(height),
        static_cast<int>(width),
        texture_size
    );
    run_pba_2d(
        reinterpret_cast<short2 *>(tex0.data_ptr<int16_t>()),
        reinterpret_cast<short2 *>(tex1.data_ptr<int16_t>()),
        reinterpret_cast<short2 *>(margin.data_ptr<int16_t>()),
        texture_size,
        stream
    );
    return tex1.index({
        torch::indexing::Slice(1, height + 1),
        torch::indexing::Slice(1, width + 1),
        torch::indexing::Slice()
    }).contiguous();
}

std::vector<torch::Tensor> cuda_distance_pair(
    torch::Tensor negative_mask,
    torch::Tensor positive_mask
) {
    TORCH_CHECK(
        negative_mask.sizes() == positive_mask.sizes(),
        "negative_mask and positive_mask must share the same shape."
    );
    return {distance_impl(negative_mask), distance_impl(positive_mask)};
}

std::vector<torch::Tensor> cuda_bbox_pair(
    torch::Tensor negative_mask,
    torch::Tensor positive_mask
) {
    TORCH_CHECK(negative_mask.is_cuda(), "negative_mask must be CUDA.");
    TORCH_CHECK(positive_mask.is_cuda(), "positive_mask must be CUDA.");
    TORCH_CHECK(negative_mask.dim() == 2, "negative_mask must be 2D.");
    TORCH_CHECK(positive_mask.dim() == 2, "positive_mask must be 2D.");
    TORCH_CHECK(
        negative_mask.sizes() == positive_mask.sizes(),
        "negative_mask and positive_mask must share the same shape."
    );

    auto guard = c10::cuda::CUDAGuard(negative_mask.device());
    auto negative_bool = negative_mask.to(torch::kBool).contiguous();
    auto positive_bool = positive_mask.to(torch::kBool).contiguous();
    auto bbox_pair = compute_bbox_pair(negative_bool, positive_bool);
    auto negative_bbox_host = bbox_to_host(bbox_pair[0]);
    auto positive_bbox_host = bbox_to_host(bbox_pair[1]);
    return {
        normalize_bbox_tensor(bbox_pair[0], negative_bbox_host),
        normalize_bbox_tensor(bbox_pair[1], positive_bbox_host),
    };
}

std::vector<torch::Tensor> cuda_sample_from_error_pair(
    torch::Tensor negative_mask,
    torch::Tensor positive_mask,
    double sfc_inner_k,
    int64_t random_value,
    bool need_debug
) {
    return sample_from_error_pair_impl(
        negative_mask,
        positive_mask,
        sfc_inner_k,
        random_value,
        need_debug
    );
}

std::vector<torch::Tensor> cuda_sample_from_masks(
    torch::Tensor pred_mask,
    torch::Tensor gt_mask,
    torch::Tensor ignore_mask,
    double sfc_inner_k,
    int64_t random_value,
    bool need_debug
) {
    TORCH_CHECK(pred_mask.is_cuda(), "pred_mask must be CUDA.");
    TORCH_CHECK(gt_mask.is_cuda(), "gt_mask must be CUDA.");
    TORCH_CHECK(ignore_mask.is_cuda(), "ignore_mask must be CUDA.");
    TORCH_CHECK(pred_mask.dim() == 2, "pred_mask must be 2D.");
    TORCH_CHECK(gt_mask.dim() == 2, "gt_mask must be 2D.");
    TORCH_CHECK(ignore_mask.dim() == 2, "ignore_mask must be 2D.");
    TORCH_CHECK(pred_mask.sizes() == gt_mask.sizes(), "pred_mask and gt_mask must match.");
    TORCH_CHECK(pred_mask.sizes() == ignore_mask.sizes(), "ignore_mask must match.");

    auto guard = c10::cuda::CUDAGuard(pred_mask.device());
    auto pred_bool = pred_mask.to(torch::kBool).contiguous();
    auto gt_bool = gt_mask.to(torch::kBool).contiguous();
    auto ignore_bool = ignore_mask.to(torch::kBool).contiguous();

    auto false_negative = torch::empty_like(pred_bool);
    auto false_positive = torch::empty_like(pred_bool);
    int numel = static_cast<int>(pred_bool.numel());
    int block = 256;
    int grid = (numel + block - 1) / block;
    auto stream = at::cuda::getDefaultCUDAStream(pred_bool.get_device());
    build_error_masks_kernel<<<grid, block, 0, stream>>>(
        pred_bool.data_ptr<bool>(),
        gt_bool.data_ptr<bool>(),
        ignore_bool.data_ptr<bool>(),
        false_negative.data_ptr<bool>(),
        false_positive.data_ptr<bool>(),
        numel
    );

    return sample_from_error_pair_impl(
        false_negative,
        false_positive,
        sfc_inner_k,
        random_value,
        need_debug
    );
}

std::vector<torch::Tensor> cuda_sample_from_masks_batch(
    torch::Tensor pred_mask,
    torch::Tensor gt_mask,
    torch::Tensor ignore_mask,
    double sfc_inner_k,
    torch::Tensor random_values
) {
    TORCH_CHECK(pred_mask.is_cuda(), "pred_mask must be CUDA.");
    TORCH_CHECK(gt_mask.is_cuda(), "gt_mask must be CUDA.");
    TORCH_CHECK(ignore_mask.is_cuda(), "ignore_mask must be CUDA.");
    TORCH_CHECK(pred_mask.dim() == 3, "pred_mask must be 3D BHW.");
    TORCH_CHECK(gt_mask.dim() == 3, "gt_mask must be 3D BHW.");
    TORCH_CHECK(ignore_mask.dim() == 3, "ignore_mask must be 3D BHW.");
    TORCH_CHECK(pred_mask.sizes() == gt_mask.sizes(), "pred_mask and gt_mask must match.");
    TORCH_CHECK(pred_mask.sizes() == ignore_mask.sizes(), "ignore_mask must match.");

    auto guard = c10::cuda::CUDAGuard(pred_mask.device());
    auto pred_bool = pred_mask.to(torch::kBool).contiguous();
    auto gt_bool = gt_mask.to(torch::kBool).contiguous();
    auto ignore_bool = ignore_mask.to(torch::kBool).contiguous();
    auto random_cpu = random_values.to(torch::kCPU, torch::kInt64).contiguous();

    int64_t batch = pred_bool.size(0);
    TORCH_CHECK(random_cpu.numel() == batch, "random_values must have B elements.");

    auto result = torch::empty(
        {batch, 3},
        torch::TensorOptions().device(pred_bool.device()).dtype(torch::kInt32)
    );
    auto meta = torch::empty(
        {batch, 4},
        torch::TensorOptions().device(pred_bool.device()).dtype(torch::kFloat32)
    );

    const int64_t *random_ptr = random_cpu.data_ptr<int64_t>();
    for (int64_t idx = 0; idx < batch; ++idx) {
        auto payload = cuda_sample_from_masks(
            pred_bool[idx],
            gt_bool[idx],
            ignore_bool[idx],
            sfc_inner_k,
            random_ptr[idx],
            false
        );
        result[idx].copy_(payload[0]);
        meta[idx].copy_(payload[1]);
    }

    return {result, meta};
}
