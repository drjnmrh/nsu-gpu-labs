#ifndef __INC_ASSETS_HPP__


#include <inttypes.h>

#include <string>
#include <unordered_map>

#include <png.h>

#include "global.hpp"


struct AssetData {
    byte* data;
    u32 size;
};


class AssetsManager {
public:
    static constexpr u32 cMaxPath = 512;

   ~AssetsManager() noexcept;

    RCode Setup() noexcept;

    RCode Load(AssetData& output, const char* assetName) noexcept;
    RCode Save(const AssetData& input, const char* assetName) const noexcept;

private:
    std::string _folder, _outpath;
    std::unordered_map<std::string, AssetData> _loaded;
};


class Png {
    Png(const Png&) = delete;
    Png& operator = (const Png&) = delete;
public:

    enum class ColorFormat {
        Undefined = 0
    ,   R8G8B8A8
    ,   R8G8B8
    ,   R8G8B8X8
    ,   R4G4B4A4
    ,   R5G6B5
    ,   A8
    };

    static u32 get_channels_number(ColorFormat format) noexcept;

    Png() noexcept;
   ~Png() noexcept {}

    Png(Png&&) noexcept;
    Png& operator = (Png&&) noexcept;

    bool IsReady() const noexcept { return _data.get() != nullptr; }

    Png Clone() const noexcept;

    RCode Load(const AssetData& ad) noexcept;
    RCode Save(AssetData& ad) const noexcept;

    RCode Convert(ColorFormat desired) noexcept;

    ColorFormat Format () const noexcept { return _format; }
    u32         Width  () const noexcept { return _width; }
    u32         Height () const noexcept { return _height; }
    u32         RowSize() const noexcept { return _szRow; }
    byte*       Data   () const noexcept { return _data.get(); }

private:

    static void swap(Png& a, Png& b) noexcept;

    static RCode convert_A8_to_RGBA(Png& dest, const Png& source) noexcept;
    static RCode convert_RGB_to_A8(Png& dest, const Png& source) noexcept;
    static RCode convert(Png& dest, const Png& source, ColorFormat target) noexcept;

    std::unique_ptr<byte[]> _data;

    u32 _szData;
    u32 _width;
    u32 _height;
    u32 _szRow;

    ColorFormat _format;
};


#define __INC_ASSETS_HPP__
#endif
