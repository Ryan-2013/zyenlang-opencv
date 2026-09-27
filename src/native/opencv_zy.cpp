#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "zyenlang_c_abi.h"

#define ZY_OPENCV_TYPE "zyenlang.opencv.Image"

static thread_local std::string zy_opencv_error = "OpenCV operation failed";

static void zy_opencv_set_error(const char* message) {
    zy_opencv_error = message ? message : "OpenCV operation failed";
}

static void zy_opencv_set_exception(const std::exception& error) {
    zy_opencv_error = error.what();
}

extern "C" ZL_String zy_opencv_last_error(void) {
    return zl_string_borrow_n(zy_opencv_error.data(), zy_opencv_error.size());
}

extern "C" void zy_opencv_image_drop(void* raw) {
    delete static_cast<cv::Mat*>(raw);
}

static cv::Mat* zy_opencv_data(ZL_Handle image) {
    return static_cast<cv::Mat*>(zl_handle_data(image, ZY_OPENCV_TYPE));
}

static ZL_Handle zy_opencv_wrap(cv::Mat* image) {
    if (image == nullptr) return ZL_Handle{};
    return zl_handle_adopt(image, ZY_OPENCV_TYPE, zy_opencv_image_drop);
}

extern "C" ZL_Handle zy_opencv_load(ZL_String path, bool color) {
    try {
        std::string file(zl_string_data(path), zl_string_byte_len(path));
        cv::Mat loaded = cv::imread(file, color ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE);
        if (loaded.empty()) {
            zy_opencv_set_error("OpenCV could not decode the image");
            return ZL_Handle{};
        }
        return zy_opencv_wrap(new cv::Mat(std::move(loaded)));
    } catch (const std::exception& error) {
        zy_opencv_set_exception(error);
        return ZL_Handle{};
    }
}

extern "C" ZL_Handle zy_opencv_from_bytes(ZL_Slice pixels, int32_t width, int32_t height, int32_t channels) {
    try {
        if (width <= 0 || height <= 0 || (channels != 1 && channels != 3 && channels != 4)) {
            zy_opencv_set_error("OpenCV image dimensions or channel count are invalid");
            return ZL_Handle{};
        }
        uint64_t expected = static_cast<uint64_t>(width) * static_cast<uint64_t>(height) * static_cast<uint64_t>(channels);
        if (pixels.element_size != sizeof(uint8_t) || pixels.len != expected || pixels.data == nullptr) {
            zy_opencv_set_error("pixel List length does not match the image shape");
            return ZL_Handle{};
        }
        int type = channels == 1 ? CV_8UC1 : channels == 3 ? CV_8UC3 : CV_8UC4;
        cv::Mat view(height, width, type, const_cast<void*>(pixels.data));
        return zy_opencv_wrap(new cv::Mat(view.clone()));
    } catch (const std::exception& error) {
        zy_opencv_set_exception(error);
        return ZL_Handle{};
    }
}

extern "C" ZL_Handle zy_opencv_clone(ZL_Handle image) {
    try {
        return zy_opencv_wrap(new cv::Mat(zy_opencv_data(image)->clone()));
    } catch (const std::exception& error) {
        zy_opencv_set_exception(error);
        return ZL_Handle{};
    }
}

extern "C" int32_t zy_opencv_width(ZL_Handle image) { return zy_opencv_data(image)->cols; }
extern "C" int32_t zy_opencv_height(ZL_Handle image) { return zy_opencv_data(image)->rows; }
extern "C" int32_t zy_opencv_channels(ZL_Handle image) { return zy_opencv_data(image)->channels(); }

extern "C" int32_t zy_opencv_save(ZL_Handle image, ZL_String path) {
    try {
        std::string file(zl_string_data(path), zl_string_byte_len(path));
        if (!cv::imwrite(file, *zy_opencv_data(image))) {
            zy_opencv_set_error("OpenCV could not encode the image");
            return -1;
        }
        return 0;
    } catch (const std::exception& error) {
        zy_opencv_set_exception(error);
        return -1;
    }
}

extern "C" ZL_Handle zy_opencv_resize(ZL_Handle image, int32_t width, int32_t height) {
    try {
        if (width <= 0 || height <= 0) {
            zy_opencv_set_error("resize dimensions must be positive");
            return ZL_Handle{};
        }
        cv::Mat* result = new cv::Mat();
        cv::resize(*zy_opencv_data(image), *result, cv::Size(width, height));
        return zy_opencv_wrap(result);
    } catch (const std::exception& error) {
        zy_opencv_set_exception(error);
        return ZL_Handle{};
    }
}

extern "C" ZL_Handle zy_opencv_grayscale(ZL_Handle image) {
    try {
        cv::Mat* source = zy_opencv_data(image);
        if (source->channels() == 1) return zy_opencv_wrap(new cv::Mat(source->clone()));
        int conversion = source->channels() == 3 ? cv::COLOR_BGR2GRAY :
            source->channels() == 4 ? cv::COLOR_BGRA2GRAY : -1;
        if (conversion < 0) {
            zy_opencv_set_error("grayscale supports one, three, or four channels");
            return ZL_Handle{};
        }
        cv::Mat* result = new cv::Mat();
        cv::cvtColor(*source, *result, conversion);
        return zy_opencv_wrap(result);
    } catch (const std::exception& error) {
        zy_opencv_set_exception(error);
        return ZL_Handle{};
    }
}

extern "C" ZL_Handle zy_opencv_gaussian_blur(ZL_Handle image, int32_t kernel_size) {
    try {
        if (kernel_size <= 0 || kernel_size % 2 == 0) {
            zy_opencv_set_error("Gaussian blur kernel size must be a positive odd number");
            return ZL_Handle{};
        }
        cv::Mat* result = new cv::Mat();
        cv::GaussianBlur(*zy_opencv_data(image), *result, cv::Size(kernel_size, kernel_size), 0.0);
        return zy_opencv_wrap(result);
    } catch (const std::exception& error) {
        zy_opencv_set_exception(error);
        return ZL_Handle{};
    }
}

extern "C" ZL_Handle zy_opencv_crop(ZL_Handle image, int32_t x, int32_t y, int32_t width, int32_t height) {
    try {
        cv::Mat* source = zy_opencv_data(image);
        if (x < 0 || y < 0 || width <= 0 || height <= 0 || x > source->cols - width || y > source->rows - height) {
            zy_opencv_set_error("crop rectangle is outside the image");
            return ZL_Handle{};
        }
        return zy_opencv_wrap(new cv::Mat((*source)(cv::Rect(x, y, width, height)).clone()));
    } catch (const std::exception& error) {
        zy_opencv_set_exception(error);
        return ZL_Handle{};
    }
}

extern "C" int32_t zy_opencv_copy_bytes(ZL_Handle image, ZL_MutSlice output) {
    try {
        cv::Mat* source = zy_opencv_data(image);
        cv::Mat contiguous = source->isContinuous() ? *source : source->clone();
        size_t byte_count = contiguous.total() * contiguous.elemSize();
        if (output.element_size != sizeof(uint8_t) || output.len != byte_count || (byte_count > 0 && output.data == nullptr)) {
            zy_opencv_set_error("output List length does not match image byte length");
            return -1;
        }
        if (byte_count > 0) std::memcpy(output.data, contiguous.data, byte_count);
        return 0;
    } catch (const std::exception& error) {
        zy_opencv_set_exception(error);
        return -1;
    }
}
