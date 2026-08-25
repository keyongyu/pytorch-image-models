// Open-set classification with an exported ncnn model — reference C++ client.
//
// Mirrors nptools/predict/predict_ncnn.py. Reads openset.ncnn.param/.bin plus the class-map file, and for
// each image prints the detected class or "unknown".
//
// Build (needs ncnn and OpenCV development packages):
//
//   g++ -O2 -std=c++17 predict_ncnn.cpp -o predict_ncnn \
//       $(pkg-config --cflags --libs opencv4) -lncnn -fopenmp
//
//   # or, if ncnn is installed somewhere non-standard:
//   g++ -O2 -std=c++17 predict_ncnn.cpp -o predict_ncnn \
//       -I/path/to/ncnn/include/ncnn -L/path/to/ncnn/lib -lncnn \
//       $(pkg-config --cflags --libs opencv4) -fopenmp
//
// Run:
//   ./predict_ncnn <run>/openset.ncnn.param <run>/openset.ncnn.bin \
//                  posmlvx/class_84.txt photo.jpg [img_size]
//
// PREPROCESSING CONTRACT — every line of it matters, and none of it fails loudly if wrong:
//
//   * BGR, i.e. cv::imread's own order. Do NOT cvtColor to RGB: the model is trained on BGR.
//   * squash resize to img_size x img_size, ignoring aspect ratio (training used
//     --crop-mode=squash), with cv::INTER_AREA. INTER_AREA averages every source pixel;
//     INTER_LINEAR (OpenCV's default, and what ncnn's own from_pixels_resize does) reads a fixed
//     2x2 kernel and aliases badly on a large downscale, which shifts the distances.
//   * mean/std 0.5 on 0..1 == substract_mean_normalize({127.5}, {1/127.5}) on 0..255 pixels.
//
// This assumes the --no-argmin export: the graph emits per-class `dists` (out0) and `margins`
// (out1) and the decision is made here. That is the variant a stock ncnn build can run; the
// in-graph-argmin variant needs layers ncnn does not ship (ArgMin, Gather).
//
// The threshold is NOT needed here: it is already baked into `margins` (margins = dists -
// threshold), so the client only checks the sign at the nearest class. That keeps this code in
// lockstep with whatever was baked into the graph.

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <net.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

// One class name per line, blank lines skipped. This is the SAME file passed to openset.py, whose
// order is the exported column order — the sidecar's "class_names" is exactly these lines.
std::vector<std::string> read_class_names(const std::string& path)
{
    std::vector<std::string> names;
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "cannot open class-map %s\n", path.c_str());
        return names;
    }
    std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n' ||
                                 line.back() == ' ' || line.back() == '\t'))
            line.pop_back();       // tolerate CRLF and trailing blanks
        if (!line.empty()) names.push_back(line);
    }
    return names;
}

struct Prediction {
    int best = -1;
    float dist = 0.f;
    float margin = 0.f;
    bool unknown = true;
};

Prediction predict(ncnn::Net& net, const cv::Mat& bgr, int img_size)
{
    cv::Mat resized;
    // squash to a square, INTER_AREA — see the contract above
    cv::resize(bgr, resized, cv::Size(img_size, img_size), 0, 0, cv::INTER_AREA);

    ncnn::Mat in = ncnn::Mat::from_pixels(resized.data, ncnn::Mat::PIXEL_BGR,
                                          resized.cols, resized.rows);
    // (x/255 - 0.5) / 0.5  ==  (x - 127.5) * (1/127.5)
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", in);

    ncnn::Mat dists, margins;
    if (ex.extract("out0", dists) != 0 || ex.extract("out1", margins) != 0) {
        fprintf(stderr, "extract failed — was this exported with --no-argmin?\n");
        return {};
    }

    Prediction p;
    const int c = dists.w;                       // [C] per-class cosine distance
    const float* d = dists;
    const float* m = margins;
    p.best = 0;
    for (int i = 1; i < c; i++)
        if (d[i] < d[p.best]) p.best = i;        // argmin on the client
    p.dist = d[p.best];
    p.margin = m[p.best];
    p.unknown = p.margin > 0.f;                  // threshold is already inside margins
    return p;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 5) {
        fprintf(stderr,
                "usage: %s openset.ncnn.param openset.ncnn.bin class_map.txt image [img_size]\n",
                argv[0]);
        return 1;
    }
    const char* param_path = argv[1];
    const char* bin_path = argv[2];
    const char* class_map = argv[3];
    const char* image_path = argv[4];
    // Not discoverable from the .param (pnnx emits Input without dims), so it is passed in. It must
    // match the img_size in the sidecar .meta.json — a mismatch does not error, it just degrades.
    const int img_size = (argc > 5) ? atoi(argv[5]) : 128;

    std::vector<std::string> names = read_class_names(class_map);
    if (names.empty()) return 1;

    ncnn::Net net;
    // Defaults are fine for this graph; set these explicitly so behaviour does not depend on how
    // the library was built. Vulkan stays off: the head is tiny and a GPU upload costs more.
    net.opt.use_vulkan_compute = false;
    net.opt.num_threads = 4;
    if (net.load_param(param_path) != 0) {
        fprintf(stderr, "failed to load %s\n", param_path);
        return 1;
    }
    if (net.load_model(bin_path) != 0) {
        fprintf(stderr, "failed to load %s\n", bin_path);
        return 1;
    }

    cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);   // BGR, alpha dropped
    if (bgr.empty()) {
        fprintf(stderr, "could not decode %s\n", image_path);
        return 1;
    }

    Prediction p = predict(net, bgr, img_size);
    if (p.best < 0) return 1;

    // Guard against a class-map that does not match the model: the exported column count is the
    // class-map line count, so a mismatch means labels would be silently shifted.
    if (p.best >= static_cast<int>(names.size())) {
        fprintf(stderr,
                "class-map has %zu entries but the model emitted index %d — wrong class-map for "
                "this model; labels would be wrong\n",
                names.size(), p.best);
        return 1;
    }

    printf("%s\n", image_path);
    printf("  detected: %s   dist=%.4f  margin=%+.4f\n",
           p.unknown ? "unknown" : names[p.best].c_str(), p.dist, p.margin);
    if (p.unknown)
        printf("  (nearest known was %s)\n", names[p.best].c_str());
    return 0;
}
