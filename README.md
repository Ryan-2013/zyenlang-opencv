# OpenCV for ZyenLang

An audited ABI v3 wrapper around OpenCV 4. `Image` owns a `cv::Mat` through an
ARC handle, so native memory is released when the final ZyenLang reference
leaves scope.

```powershell
zy install opencv
```

OpenCV development headers and libraries must be installed on the build
machine. Linux package managers usually provide `libopencv-dev`; macOS users
can use Homebrew `opencv`; Windows users can use an MSYS2/MinGW OpenCV package
matching the compiler used by ZyenLang. Add nonstandard include/library paths
through `ZY_CXXFLAGS` and `ZY_LDFLAGS`.

```zy
import opencv as cv

fn main() i32 throws Error {
    let image = cv::load("photo.png")
    let preview = image.resize(640, 360).gaussian_blur(5)
    preview.save("preview.png")
    return 0
}
```

## API

- `load(path, color = true)` and `from_bytes(pixels, width, height, channels)`
- `Image.width()`, `height()`, `channels()`, `byte_length()`
- `Image.clone()`, `resize()`, `grayscale()`, `gaussian_blur()`, `crop()`
- `Image.save()` and `to_bytes()`

The adapter catches C++ exceptions at the ABI boundary and converts them into
ZyenLang `Error` values. It never exposes `cv::Mat*` or raw pointers.
