#include <iostream>

#include "global.hpp"

#include "assets.cpp"


int main(void) {

    AssetsManager am;
    RCode rc = am.Setup();
    if (rc != RCode::Ok) {
        std::cerr << "Failed to setup Assets Manager (" << (int)rc << ")\n";
        return static_cast<int>(rc);
    }

    AssetData ad;
    rc = am.Load(ad, "stonedfox_artsy.png");
    if (rc != RCode::Ok) {
        std::cerr << "Failed to load asset (" << (int)rc << ")\n";
        return static_cast<int>(rc);
    }

    Png png;
    rc = png.Load(ad);
    if (rc != RCode::Ok) {
        std::cerr << "Failed to load png (" << (int)rc << ")\n";
        return static_cast<int>(rc);
    }

    std::cout << static_cast<int>(png.Format()) << std::endl;

    rc = png.Convert(Png::ColorFormat::A8);
    if (rc != RCode::Ok) {
        std::cerr << "Failed to convert png (" << (int)rc << ")\n";
        return static_cast<int>(rc);
    }

    rc = png.Convert(Png::ColorFormat::R8G8B8A8);
    if (rc != RCode::Ok) {
        std::cerr << "Failed to convert png (" << (int)rc << ")\n";
        return static_cast<int>(rc);
    }

    rc = png.Save(ad);
    if (rc != RCode::Ok) {
        std::cerr << "Failed to save png (" << (int)rc << ")\n";
        return static_cast<int>(rc);
    }

    rc = am.Save(ad, "saved.png");
    if (rc != RCode::Ok) {
        std::cerr << "Failed to save asset (" << (int)rc << ")\n";
        return static_cast<int>(rc);
    }

    return CODE(Ok);
}
