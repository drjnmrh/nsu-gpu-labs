#include "assets.hpp"

#include <cstring>
#include <fstream>
#include <functional>
#include <vector>

#if defined(_WIN32) || defined(_WIN64) || defined(WIN32) || defined(WIN64) || defined(__CYGWIN__)
#   include <direct.h>
#   define getcwd _getcwd
#else
#   include <unistd.h>
#endif


namespace {


    template <int...> struct seq {};

    template <int N, int... S> struct gens : gens<N - 1, N - 1, S...> {};
    template <int... S> struct gens<0, S...> { typedef seq<S...> type; };

    template<typename... Params>
    struct __raii_t {
        using Deleter = std::function<void(Params *...)>;

        __raii_t(Params *...params, Deleter deleter) noexcept
            : _data(params...), _deleter(deleter), _released(false)
        {}

        __raii_t(__raii_t && another) = default;

        ~__raii_t() noexcept {
            if (!_released) {
                doDelete(typename gens<sizeof...(Params)>::type());
            }
        }

        void release() noexcept { _released = true; }

        template <int... S> void doDelete(seq<S...>) noexcept {
            _deleter(std::get<S>(_data)...);
        }

        std::tuple<Params *...> _data;
        std::function<void(Params *...)> _deleter;
        bool _released;
    };

    template <typename... Params>
    static __raii_t<Params...> __ToRaii(typename __raii_t<Params...>::Deleter deleter, Params *...params) noexcept {
        return __raii_t<Params...>(params..., deleter);
    }

    typedef struct {
        size_t offset;
        const byte*const* ppData;
    } read_data_t;

    static void read_png_data(png_structp pngPtr, png_bytep data, png_size_t len) {
        read_data_t& rd = *(read_data_t*)png_get_io_ptr(pngPtr);
        std::memcpy(data, *rd.ppData + rd.offset, len);
        rd.offset += len;
    }

    typedef struct {
        size_t size;
        byte* pData;
    } write_data_t;

    static void write_png_data(png_structp pngPtr, png_bytep data, png_size_t len) {
        write_data_t& wd = *(write_data_t*)png_get_io_ptr(pngPtr);
        std::size_t sz = wd.size + len;
        if (!wd.pData) {
            wd.pData = (byte*)std::malloc(sz);
        } else {
            wd.pData = (byte*)std::realloc(wd.pData, sz);
        }
        if(!wd.pData) {
            png_error(pngPtr, "failed to allocate buffer");
            return;
        }

        std::memcpy(wd.pData + wd.size, data, len);
        wd.size = sz;
    }


    class ColorObject {
    public:

        static
        ColorObject FromFormat(Png::ColorFormat format) noexcept;

        ColorObject(u32 nbBitsR, u32 nbBitsG, u32 nbBitsB, u32 nbBitsA) noexcept;

        bool isValid() const noexcept { return _nbBits > 0 && _nbBits <= 32; }

        u32 size() const noexcept { return _nbBytes; }

        byte R() const noexcept { return _r; }
        byte G() const noexcept { return _g; }
        byte B() const noexcept { return _b; }
        byte A() const noexcept { return _a; }

        void set(byte r, byte g, byte b, byte a) noexcept;

        RCode save(byte* pBuffer) const noexcept;
        RCode load(const byte* pBuffer) noexcept;

    private:

        u32 _nbBitsR;
        u32 _nbBitsG;
        u32 _nbBitsB;
        u32 _nbBitsA;

        u32 _nbBits;
        u32 _nbBytes;

        byte _r, _g, _b, _a;
    };

    /*static*/
    ColorObject ColorObject::FromFormat(Png::ColorFormat format) noexcept {

        using ColorFormat = Png::ColorFormat;

        switch (format) {
            case ColorFormat::R8G8B8X8 : return ColorObject(8, 8, 8, 8);
            case ColorFormat::R8G8B8A8 : return ColorObject(8, 8, 8, 8);
            case ColorFormat::R8G8B8   : return ColorObject(8, 8, 8, 0);
            case ColorFormat::R4G4B4A4 : return ColorObject(4, 4, 4, 4);
            case ColorFormat::R5G6B5   : return ColorObject(5, 6, 5, 0);
            case ColorFormat::A8       : return ColorObject(0, 0, 0, 8);
            case ColorFormat::Undefined: return ColorObject(0, 0, 0, 0);
        }

        return ColorObject(0, 0, 0, 0);
    }


    ColorObject::ColorObject(u32 nbBitsR, u32 nbBitsG, u32 nbBitsB, u32 nbBitsA) noexcept
        : _nbBitsR(nbBitsR), _nbBitsG(nbBitsG), _nbBitsB(nbBitsB), _nbBitsA(nbBitsA)
        , _nbBits(nbBitsA + nbBitsB + nbBitsG + nbBitsR)
        , _r(0), _g(0), _b(0), _a(0)
    {
        _nbBytes = _nbBits / 8;
        if ((_nbBits % 8) != 0) {
            _nbBytes += 1;
        }
    }


    void ColorObject::set(byte r, byte g, byte b, byte a) noexcept {

        u32 maxR = (1 << _nbBitsR) - 1;
        u32 maxG = (1 << _nbBitsG) - 1;
        u32 maxB = (1 << _nbBitsB) - 1;
        u32 maxA = (1 << _nbBitsA) - 1;

        _r = static_cast<byte>(std::min((u32)r, maxR));
        _g = static_cast<byte>(std::min((u32)g, maxG));
        _b = static_cast<byte>(std::min((u32)b, maxB));
        _a = static_cast<byte>(std::min((u32)a, maxA));
    }


    RCode ColorObject::save(byte* pBuffer) const noexcept {

        if (!isValid()) {
            return RCode::InvalidInput;
        }

        assert(_nbBits <= 32);

        if (_nbBits <= 8) {
            byte colorval = (_r << (_nbBits-_nbBitsR))
                          | (_g << (_nbBitsB + _nbBitsA))
                          | (_b << _nbBitsA)
                          | (_a);
            *pBuffer = colorval;
        } else if (_nbBits <= 16) {
            u16 colorval = (static_cast<u16>(_r) << (_nbBits-_nbBitsR))
                         | (static_cast<u16>(_g) << (_nbBitsB + _nbBitsA))
                         | (static_cast<u16>(_b) << _nbBitsA)
                         | static_cast<u16>(_a);
            *reinterpret_cast<u16*>(pBuffer) = colorval;
        } else {
            u32 colorval = (static_cast<u32>(_r) << (_nbBits-_nbBitsR))
                         | (static_cast<u32>(_g) << (_nbBitsB + _nbBitsA))
                         | (static_cast<u32>(_b) << _nbBitsA)
                         | static_cast<u32>(_a);
            *reinterpret_cast<u32*>(pBuffer) = colorval;
        }

        return RCode::Ok;
    }


    RCode ColorObject::load(const byte* pBuffer) noexcept {

        if (!isValid()) {
            return RCode::InvalidInput;
        }

        assert(_nbBits <= 32);

        u32 maxG = (1 << _nbBitsG) - 1;
        u32 maxB = (1 << _nbBitsB) - 1;
        u32 maxA = (1 << _nbBitsA) - 1;

        u32 colorval;
        if (_nbBits <= 8) {
            colorval = static_cast<u32>(*pBuffer);
        } else if (_nbBits <= 16) {
            colorval = static_cast<u32>(*reinterpret_cast<const u16*>(pBuffer));
        } else {
            colorval = *reinterpret_cast<const u32*>(pBuffer);
        }

        byte rv = _nbBitsR > 0 ? static_cast<byte>(colorval >> (_nbBits-_nbBitsR)) : 0;
        byte gv = _nbBitsG > 0 ? static_cast<byte>((colorval >> (_nbBitsB + _nbBitsA)) & maxG) : 0;
        byte bv = _nbBitsB > 0 ? static_cast<byte>((colorval >> _nbBitsA) & maxB) : 0;
        byte av = _nbBitsA > 0 ? static_cast<byte>(colorval & maxA) : 255;
        set(rv, gv, bv, av);

        return RCode::Ok;
    }


}


/*static*/
u32 Png::get_channels_number(ColorFormat format) noexcept {

    switch(format) {
        case ColorFormat::Undefined: return 0;
        case ColorFormat::R8G8B8A8 : return 4;
        case ColorFormat::R8G8B8   : return 3;
        case ColorFormat::R8G8B8X8 : return 3;
        case ColorFormat::R4G4B4A4 : return 4;
        case ColorFormat::R5G6B5   : return 3;
        case ColorFormat::A8       : return 1;
        default: return 0;
    }
}


Png::Png() noexcept
: _data(nullptr)
, _szData(0)
, _width(0), _height(0)
, _szRow(0)
, _format(Png::ColorFormat::Undefined)
{}


Png::Png(Png&& another) noexcept
    : _data(another._data.release())
    , _szData(another._szData)
    , _width(another._width), _height(another._height)
    , _szRow(another._szRow)
    , _format(another._format)
{}


Png& Png::operator = (Png&& another) noexcept {

    if (this == &another) {
        return *this;
    }

    swap(*this, another);

    return *this;
}


Png Png::Clone() const noexcept {

    Png result;
    if (!IsReady()) {
        return result;
    }

    result._data = std::make_unique<byte[]>(_szData);
    if (!result._data.get()) {
        return result;
    }

    std::memcpy(result._data.get(), _data.get(), _szData);

    result._szData = _szData;
    result._width = _width;
    result._height = _height;
    result._szRow = _szRow;
    result._format = _format;

    return result;
}


RCode Png::Load(const AssetData& ad) noexcept {

    if (!ad.data) {
        return RCode::InvalidInput;
    }

    if (png_sig_cmp(ad.data, 0, 8)) {
        return RCode::InvalidInput;
    }

    read_data_t rd{8, &ad.data};

    png_structp pPng =
        png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!pPng) {
        return RCode::MemError;
    }
    auto raii__pPng = __ToRaii([](png_structp* p) {
        png_destroy_read_struct(p, nullptr, nullptr);
    }, &pPng);

    png_infop pPngInfo = png_create_info_struct(pPng);
    if (!pPngInfo) {
       return RCode::MemError;
    }
    raii__pPng.release();
    auto raii__pPngInfo = __ToRaii([](png_structp* p1, png_infop* p2){
        png_destroy_read_struct(p1, p2, nullptr);
    }, &pPng, &pPngInfo);

    png_uint_32 w, h;
    int bDepth, colorType, interlaceType;
    ColorFormat fmt = ColorFormat::Undefined;
    png_size_t szRowInBytes = 0;
    std::vector<uint8_t*> rows;

    if (setjmp(png_jmpbuf(pPng))) {
        return RCode::LogicError;
    }

    png_set_read_fn(pPng, &rd, read_png_data);

    png_set_sig_bytes(pPng, 8);

    // the call to @ref png_read_info() gives us all of the information from the
    // PNG file before the first IDAT (image data chunk)
    png_read_info(pPng, pPngInfo);

    // get PNG_IHDR chunk information from png_info structure
    png_get_IHDR(pPng, pPngInfo, &w, &h, &bDepth, &colorType, &interlaceType, NULL, NULL);

    rows.resize(static_cast<std::size_t>(h));
    std::memset(rows.data(), 0, rows.size() * sizeof(rows[0]));

    // strip the multibyte color channel values to single-byte to avoid bit-order mess
    // (png reads for all platforms bytes as big-endian, mean while, storing these bytes
    // inside the custom buffer and further interpretation will be done in a platform
    // specific endianness, thus it's easier to just strip multibytes into the singlebytes).
    png_set_strip_16(pPng);

    // convert the grayscale image to the RGBA 8 bit image
    if (colorType == PNG_COLOR_TYPE_GRAY_ALPHA) {
        fmt = ColorFormat::R8G8B8A8;
        png_set_gray_to_rgb(pPng);
    }

    // extract multiple pixels with bit depths of 1, 2, and 4 from a single
    // byte into separate bytes (useful for paletted and grayscale images)
    png_set_packing(pPng);

    // expand paletted colors into true RGB triplets
    if (colorType == PNG_COLOR_TYPE_PALETTE) {
        fmt = ColorFormat::R8G8B8;
        png_set_palette_to_rgb(pPng);
    }

    // expand grayscale images to the full 8 bits from 1, 2, or 4 bits/pixel
    if (colorType == PNG_COLOR_TYPE_GRAY && bDepth < 8) {
        fmt = ColorFormat::A8;
        png_set_expand_gray_1_2_4_to_8(pPng);
    }

    // expand paletted or RGB images with transparency to full alpha channels
    // so the data will be available as RGBA quartets
    if (png_get_valid(pPng, pPngInfo, PNG_INFO_tRNS) != 0) {
        png_set_tRNS_to_alpha(pPng);
    }

    // set the background color to draw transparent and alpha images over.
    // It is possible to set the red, green, and blue components directly
    // for paletted images instead of supplying a palette index. Note that
    // even if the PNG file supplies a background, you are not required to
    // use it - you should use the (solid) application background if it has one.
    if (colorType != PNG_COLOR_TYPE_RGBA) {
        png_color_16 myBackground, *pImageBackground;

        if (png_get_bKGD(pPng, pPngInfo, &pImageBackground) != 0) {
            png_set_background(pPng, pImageBackground, PNG_BACKGROUND_GAMMA_FILE, 1, 1.0);
        } else {
            png_set_background(pPng, &myBackground, PNG_BACKGROUND_GAMMA_SCREEN, 0, 1.0);
        }
    }

    /* Optional call to gamma correct and add the background to the palette
     * and update info structure. REQUIRED if you are expecting libpng to
     * update the palette for you (ie you selected such a transform above). */
    png_read_update_info(pPng, pPngInfo);

    szRowInBytes = png_get_rowbytes(pPng, pPngInfo);

    _szData = static_cast<u32>(h*szRowInBytes);
    std::unique_ptr<byte[]> data = std::make_unique<byte[]>(_szData);
    if (!data.get()) {
        return RCode::MemError;
    }

    for (png_uint_32 r = 0; r < h; r++) {
        rows[h - 1 - r] = data.get() + szRowInBytes * r;
    }

    png_read_image(pPng, rows.data());

    // read rest of file, and get additional chunks in pPngInfo
    png_read_end(pPng, pPngInfo);

    raii__pPngInfo.release();
    png_destroy_read_struct(&pPng, &pPngInfo, NULL);

    if (ColorFormat::Undefined == fmt) {
        switch (colorType) {
            case PNG_COLOR_TYPE_GRAY     : fmt = ColorFormat::A8;       break;
            case PNG_COLOR_TYPE_RGB      : fmt = ColorFormat::R8G8B8;   break;
            case PNG_COLOR_TYPE_RGB_ALPHA: fmt = ColorFormat::R8G8B8A8; break;
            default: fmt = ColorFormat::Undefined;
        }
    }

    if (fmt == ColorFormat::Undefined) {
        return RCode::InvalidInput;
    }

    if (8 != bDepth) {
        return RCode::InvalidInput;
    }

    _data = std::move(data);

    _width = static_cast<u32>(w);
    _height = static_cast<u32>(h);

    _format = fmt;

    _szRow = static_cast<u32>(szRowInBytes);

    return RCode::Ok;
}


RCode Png::Save(AssetData& ad) const noexcept {

    if (!IsReady()) {
        return RCode::InvalidInput;
    }

    png_structp pPng =
        png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!pPng) {
        return RCode::MemError;
    }
    auto raii__pPng = __ToRaii([](png_structp* p) {
        png_destroy_write_struct(p, nullptr);
    }, &pPng);

    png_infop pPngInfo = png_create_info_struct(pPng);
    if (!pPngInfo) {
       return RCode::MemError;
    }
    raii__pPng.release();
    auto raii__pPngInfo = __ToRaii([](png_structp* p1, png_infop* p2){
        png_destroy_write_struct(p1, p2);
    }, &pPng, &pPngInfo);

    if (setjmp(png_jmpbuf(pPng))) {
        return RCode::LogicError;
    }

    write_data_t wd{0, nullptr};

    Png copy = Clone();

    int colortype = -1;
    switch(_format) {
        case ColorFormat::R8G8B8A8: colortype = PNG_COLOR_TYPE_RGB_ALPHA; break;
        case ColorFormat::R8G8B8  :
        case ColorFormat::R8G8B8X8: colortype = PNG_COLOR_TYPE_RGB; break;
        case ColorFormat::R4G4B4A4:
        case ColorFormat::R5G6B5: {
            copy.Convert(ColorFormat::R8G8B8A8);
            colortype = PNG_COLOR_TYPE_RGB_ALPHA; break;
        } break;
        case ColorFormat::A8: colortype = PNG_COLOR_TYPE_GRAY; break;
        default: break;
    }

    if (colortype < 0) {
        return RCode::LogicError;
    }

    png_set_IHDR(pPng, pPngInfo, _width, _height, 8, colortype,
        PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    std::vector<byte*> rows(_height);
    for (png_uint_32 r = 0; r < _height; r++) {
        rows[_height - 1 - r] = _data.get() + _szRow * r;
    }

    png_set_rows(pPng, pPngInfo, &rows[0]);

    png_set_write_fn(pPng, &wd, write_png_data, NULL);
    png_write_png(pPng, pPngInfo, PNG_TRANSFORM_IDENTITY, NULL);

    ad.data = new byte[wd.size];
    if (!ad.data) {
        std::free(wd.pData);
        return RCode::MemError;
    }

    ad.size = wd.size;

    std::memcpy(ad.data, wd.pData, ad.size);

    std::free(wd.pData);

    return RCode::Ok;
}


RCode Png::Convert(ColorFormat desired) noexcept {

    if (desired == _format) {
        return RCode::Ok;
    }

    Png tmp;
    RCode rcode = convert(tmp, *this, desired);
    if (rcode != RCode::Ok) {
        return rcode;
    }

    swap(tmp, *this);

    return RCode::Ok;
}


/*static*/
void Png::swap(Png& a, Png& b) noexcept {

    a._data.swap(b._data);

    std::swap(a._szData, b._szData);
    std::swap(a._width, b._width);
    std::swap(a._height, b._height);
    std::swap(a._szRow, b._szRow);
    std::swap(a._format, b._format);
}


/*static*/
RCode Png::convert_A8_to_RGBA(Png& dest, const Png& source) noexcept {

    assert(source._format == ColorFormat::A8);

    ColorObject srcColor = ColorObject::FromFormat(ColorFormat::A8);
    assert(srcColor.isValid());

    ColorObject dstColor = ColorObject::FromFormat(ColorFormat::R8G8B8A8);
    assert(srcColor.isValid());

    dest._szRow = source._width * dstColor.size();
    dest._width = source._width;
    dest._height = source._height;
    dest._format = ColorFormat::R8G8B8A8;
    dest._szData = source._width * source._height * dstColor.size();
    dest._data = std::make_unique<byte[]>(dest._szData);
    if (!dest._data) {
        return RCode::MemError;
    }

    for (uint32_t y = 0, w = 0; y < source._height * source._szRow; y += source._szRow, w += dest._szRow) {
        for (uint32_t x = 0, v = 0; x < source._width * srcColor.size(); x += srcColor.size(), v += dstColor.size()) {
            assert(y + x < source._szData);
            assert(w + v < dest._szData);
            srcColor.load(&(source._data[y + x]));
            dstColor.set(srcColor.A(), srcColor.A(), srcColor.A(), 255);
            dstColor.save(&(dest._data[w + v]));
        }
    }

    return RCode::Ok;
}


/*static*/
RCode Png::convert_RGB_to_A8(Png& dest, const Png& source) noexcept {

    assert(source._format == ColorFormat::R8G8B8);

    ColorObject srcColor = ColorObject::FromFormat(ColorFormat::R8G8B8);
    assert(srcColor.isValid());

    ColorObject dstColor = ColorObject::FromFormat(ColorFormat::A8);
    assert(srcColor.isValid());

    dest._szRow = source._width * dstColor.size();
    dest._width = source._width;
    dest._height = source._height;
    dest._format = ColorFormat::A8;
    dest._szData = source._width * source._height * dstColor.size();
    dest._data = std::make_unique<byte[]>(dest._szData);
    if (!dest._data) {
        return RCode::MemError;
    }

    for (uint32_t y = 0, w = 0; y < source._height * source._szRow; y += source._szRow, w += dest._szRow) {
        for (uint32_t x = 0, v = 0; x < source._width * srcColor.size(); x += srcColor.size(), v += dstColor.size()) {
            assert(y + x < source._szData);
            assert(w + v < dest._szData);
            srcColor.load(&(source._data[y + x]));
            float clr = 0.299 * srcColor.R() + 0.587 * srcColor.G() + 0.114 * srcColor.B();
            dstColor.set(clr, clr, clr, clr);
            dstColor.save(&(dest._data[w + v]));
        }
    }

    return RCode::Ok;
}


/*static*/
RCode Png::convert(Png& dest, const Png& source, ColorFormat target) noexcept {

    if (source._format == ColorFormat::A8 && target == ColorFormat::R8G8B8A8) {
        return convert_A8_to_RGBA(dest, source);
    }

    if (source._format == ColorFormat::R8G8B8 && target == ColorFormat::A8) {
        return convert_RGB_to_A8(dest, source);
    }

    ColorObject srcColor = ColorObject::FromFormat(source._format);
    if (!srcColor.isValid()) {
        return RCode::InvalidInput;
    }

    ColorObject dstColor = ColorObject::FromFormat(target);
    if (!dstColor.isValid()) {
        return RCode::InvalidInput;
    }

    dest._szRow = source._width * dstColor.size();
    dest._width = source._width;
    dest._height = source._height;
    dest._format = target;
    dest._szData = source._width * source._height * dstColor.size();
    dest._data = std::make_unique<byte[]>(dest._szData);
    if (!dest._data.get()) {
        return RCode::MemError;
    }

    for (uint32_t y = 0, w = 0; y < source._height * source._szRow; y += source._szRow, w += dest._szRow) {
        for (uint32_t x = 0, v = 0; x < source._width * srcColor.size(); x += srcColor.size(), v += dstColor.size()) {
            assert(y + x < source._szData);
            assert(w + v < dest._szData);
            srcColor.load(&(source._data[y + x]));
            dstColor.set(srcColor.R(), srcColor.G(), srcColor.B(), srcColor.A());
            dstColor.save(&(dest._data[w + v]));
        }
    }

    return RCode::Ok;
}


AssetsManager::~AssetsManager() noexcept {

    for (auto& p : _loaded) {
        delete[] p.second.data;
    }
}


RCode AssetsManager::Setup() noexcept {

    char pathbuf[cMaxPath];
    char* res = getcwd(pathbuf, cMaxPath);
    if (!res) {
        return RCode::IOError;
    }

    _outpath = std::string(pathbuf) + std::string("/");

    static char sAssetsPath[] = "/../assets/";

    size_t l = std::strlen(pathbuf);
    if (l + lengthof(sAssetsPath) >= lengthof(pathbuf)) {
        return RCode::LogicError;
    }

    std::strcat(pathbuf, sAssetsPath);

    _folder = std::string(pathbuf);

    return RCode::Ok;
}


RCode AssetsManager::Load(AssetData& output, const char* assetName) noexcept {

    if (_folder.size() == 0) {
        return RCode::LogicError;
    }

    auto foundIt = _loaded.find(std::string(assetName));
    if (foundIt != _loaded.end()) {
        output = foundIt->second;
        return RCode::Ok;
    }

    char pathbuf[cMaxPath];
    if (std::strlen(assetName) + _folder.size() >= lengthof(pathbuf)) {
        return RCode::InvalidInput;
    }

    std::strcpy(pathbuf, _folder.c_str());
    std::strcat(pathbuf, assetName);

    try {
        std::ifstream ifs(pathbuf);
        if (!ifs.is_open()) {
            return RCode::InvalidInput;
        }

        ifs.seekg(0, std::ios::end);
        size_t length = ifs.tellg();
        ifs.seekg(0, std::ios::beg);

        output.data = new byte[length];
        if (!output.data) {
            return RCode::MemError;
        }
        output.size = length;

        ifs.read((char*)output.data, length);
    } catch (std::bad_alloc&) {
        return RCode::MemError;
    } catch (...) {
        return RCode::InvalidInput;
    }

    try {
        auto [it, res] = _loaded.insert(std::make_pair(std::string(assetName), output));
        if (!res) {
            delete[] output.data;
            return RCode::LogicError;
        }
    } catch(...) {
        delete[] output.data;
        return RCode::MemError;
    }

    return RCode::Ok;
}


RCode AssetsManager::Save(const AssetData& input, const char* assetName) const noexcept {

    if (_outpath.size() == 0) {
        return RCode::LogicError;
    }

    char pathbuf[cMaxPath];
    if (std::strlen(assetName) + _outpath.size() >= lengthof(pathbuf)) {
        return RCode::InvalidInput;
    }

    std::strcpy(pathbuf, _outpath.c_str());
    std::strcat(pathbuf, assetName);

    try {
        std::ofstream ofs(pathbuf);
        if (!ofs.is_open()) {
            return RCode::IOError;
        }

        ofs.write((const char*)input.data, input.size);
    } catch (...) {
        return RCode::IOError;
    }

    return RCode::Ok;
}
