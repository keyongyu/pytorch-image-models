// Sweep ncnn's optimisation flags on the open-set model: speed AND numerical drift.
//
//   ./opts_ncnn <run>/openset.ncnn.param <run>/openset.ncnn.bin <image-dir>
//
// RUN THIS ON THE DEPLOYMENT DEVICE. Measured on x86 (Threadripper 3960X, 1 thread, 100 crops at
// 128px) every fp16 flag was inert -- 0.000000 drift, 0% speed -- because ncnn's fp16 storage and
// arithmetic are ARM/NEON features that x86 accepts and ignores. The two flags that did matter
// there were:
//
//   use_sgemm_convolution = false   ~3.4% faster (1.92 vs 1.98 ms/image). This model is depthwise
//                                   and 1x1-pointwise heavy, so im2col+GEMM packing overhead does
//                                   not pay for kernels that small.
//   use_packing_layout    = false   66% SLOWER -- leave it on.
//
// On ARM expect fp16_storage / fp16_arithmetic to be where the win is, with a numerical shift worth
// checking against the reject threshold: the "max|d-ref|" column is how much the nearest-prototype
// distances moved versus the baseline config, and the threshold is ~0.117 (28 degrees).
//
// Flags are set BEFORE load_model on purpose -- several are baked into the packed weights at load
// time, and ncnn warns ("convolution gemm will use load-time value") if they change afterwards.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <dirent.h>
#include <net.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
using clk = std::chrono::steady_clock;
static double ms(clk::duration d) { return std::chrono::duration<double,std::milli>(d).count(); }

struct Cfg { const char* name; bool fp16s, fp16a, fp16p, wino, sgemm, pack; };

int main(int argc, char** argv)
{
    // gather up to 100 crops
    std::vector<cv::Mat> raw;
    if (DIR* d = opendir(argv[3])) {
        std::vector<std::string> ps;
        while (dirent* e = readdir(d)) {
            std::string n = e->d_name;
            if (n.find(".jpe") != std::string::npos || n.find(".jpg") != std::string::npos ||
                n.find(".png") != std::string::npos) ps.push_back(std::string(argv[3]) + "/" + n);
            if (ps.size() >= 100) break;
        }
        closedir(d);
        for (auto& p : ps) { cv::Mat m = cv::imread(p, cv::IMREAD_COLOR); if (!m.empty()) raw.push_back(m); }
    }
    const int n = (int)raw.size();
    const int img_size = 128, reps = 5;
    const float mv[3] = {127.5f,127.5f,127.5f}, nv[3] = {1/127.5f,1/127.5f,1/127.5f};

    std::vector<cv::Mat> small(n);
    for (int i = 0; i < n; i++)
        cv::resize(raw[i], small[i], cv::Size(img_size,img_size), 0, 0, cv::INTER_AREA);

    Cfg cfgs[] = {
        //                          fp16s  fp16a  fp16p  wino   sgemm  pack
        {"baseline (defaults)",     true,  false, true,  true,  true,  true },
        {"no fp16 storage",         false, false, false, true,  true,  true },
        {"fp16 arithmetic",         true,  true,  true,  true,  true,  true },
        {"no winograd",             true,  false, true,  false, true,  true },
        {"no sgemm",                true,  false, true,  true,  false, true },
        {"no packing",              true,  false, true,  true,  true,  false},
        {"fp16 arith, no winograd", true,  true,  true,  false, true,  true },
    };

    std::vector<float> ref;              // baseline dists of image 0, for drift
    printf("%-24s %10s %10s %12s\n", "config", "ms/image", "crops/s", "max|d-ref|");
    for (const Cfg& c : cfgs) {
        ncnn::Net net;
        net.opt.use_vulkan_compute = false;
        net.opt.num_threads = 1;
        net.opt.use_fp16_storage = c.fp16s;
        net.opt.use_fp16_arithmetic = c.fp16a;
        net.opt.use_fp16_packed = c.fp16p;
        net.opt.use_winograd_convolution = c.wino;
        net.opt.use_sgemm_convolution = c.sgemm;
        net.opt.use_packing_layout = c.pack;
        if (net.load_param(argv[1]) != 0 || net.load_model(argv[2]) != 0) { printf("load failed\n"); return 1; }

        double best = 1e18; std::vector<float> got;
        for (int r = 0; r < reps; r++) {
            clk::time_point t0 = clk::now();
            for (int i = 0; i < n; i++) {
                ncnn::Mat in = ncnn::Mat::from_pixels(small[i].data, ncnn::Mat::PIXEL_BGR,
                                                      small[i].cols, small[i].rows, (int)small[i].step);
                in.substract_mean_normalize(mv, nv);
                ncnn::Extractor ex = net.create_extractor();
                ex.input("in0", in);
                ncnn::Mat dd; ex.extract("out0", dd);
                if (i == 0 && r == 0) { got.assign((const float*)dd, (const float*)dd + dd.w); }
            }
            best = std::min(best, ms(clk::now() - t0));
        }
        if (ref.empty()) ref = got;
        double drift = 0;
        for (size_t k = 0; k < ref.size() && k < got.size(); k++)
            drift = std::max(drift, (double)std::abs(ref[k] - got[k]));
        printf("%-24s %10.3f %10.0f %12.6f\n", c.name, best/n, n*1000.0/best, drift);
    }
    return 0;
}
