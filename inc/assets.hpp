#ifndef __INC_ASSETS_HPP__


#include <inttypes.h>

#include "global.hpp"


struct AssetData {
    byte* data;
    u32 size;
};


enum class ColorFormat {
    Undefined = 0
,   R8G8B8A8
,   R8G8B8
,   R8G8B8X8
,   R4G4B4A4
,   R5G6B5
,   A8
};


class Png {
public:

    Png() noexcept;
   ~Png() noexcept;

    bool IsReady() const noexcept { return _data != nullptr; }

    RCode Load(const AssetData& ad) noexcept;
    RCode Convert(ColorFormat desired) noexcept;

    ColorFormat Format () const noexcept { return _format; }
    u32         Width  () const noexcept { return _width; }
    u32         Height () const noexcept { return _height; }
    u32         RowSize() const noexcept { return _szRow; }
    byte*       Data   () const noexcept { return _data; }

private:

    byte* _data;

    u32 _szData;
    u32 _width;
    u32 _height;
    u32 _szRow;

    ColorFormat _format;
};


#define __INC_ASSETS_HPP__
#endif
