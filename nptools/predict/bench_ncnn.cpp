// Throughput benchmark for the open-set ncnn model: N crops through one loaded Net.
//
// Separates the three costs a real pipeline pays per box -- resize, Mat build + normalize, forward
// -- because which one dominates decides where optimisation is worth spending.
//
//   ./bench_ncnn openset.ncnn.param openset.ncnn.bin <image-dir> [img_size] [threads] [reps] [resize]
//
// [resize] selects how a crop is scaled to img_size:
//   area   (default) cv::resize INTER_AREA, then ncnn::Mat::from_pixels -- the EXACT path, the
//                    same filter the prototypes were built with.
//   ncnn             ncnn::Mat::from_pixels_resize, folding the scale into the Mat build. ~22x
//                    cheaper for that step, but BILINEAR: on a large downscale it samples a fixed
//                    2x2 neighbourhood and aliases, so the distances shift.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include <net.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <dirent.h>

using clk = std::chrono::steady_clock;
static double ms(clk::duration d)
{
    return std::chrono::duration<double, std::milli>(d).count();
}

int main(int argc, char** argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s param bin image_dir [img_size] [threads] [reps]\n", argv[0]);
        return 1;
    }
    const int img_size = (argc > 4) ? atoi(argv[4]) : 128;
    const int threads = (argc > 5) ? atoi(argv[5]) : 4;
    const int reps = (argc > 6) ? atoi(argv[6]) : 5;
    const std::string resize_mode = (argc > 7) ? argv[7] : "area";
    if (resize_mode != "area" && resize_mode != "ncnn") {
        fprintf(stderr, "[resize] must be \"area\" or \"ncnn\", got \"%s\"\n", resize_mode.c_str());
        return 1;
    }
    const bool use_ncnn_resize = (resize_mode == "ncnn");

    // Collect up to 100 images from the directory (non-recursive is enough for a benchmark).
    std::vector<std::string> paths;
    if (DIR* d = opendir(argv[3])) {
        while (dirent* e = readdir(d)) {
            std::string n = e->d_name;
            if (n.size() > 4 && (n.rfind(".jpg") != std::string::npos ||
                                 n.rfind(".jpeg") != std::string::npos ||
                                 n.rfind(".png") != std::string::npos))
                paths.push_back(std::string(argv[3]) + "/" + n);
            if (paths.size() >= 100) break;
        }
        closedir(d);
    }
    if (paths.empty()) {
        fprintf(stderr, "no images in %s\n", argv[3]);
        return 1;
    }
    std::sort(paths.begin(), paths.end());

    // Decode once, outside the timed region: in production the detector hands over crops that are
    // already decoded, so charging JPEG decode to the classifier would overstate its cost.
    std::vector<cv::Mat> raw;
    for (const auto& p : paths) {
        cv::Mat m = cv::imread(p, cv::IMREAD_COLOR);
        if (!m.empty()) raw.push_back(m);
    }
    const int n = static_cast<int>(raw.size());

    // num_threads set BEFORE load: ncnn's gemm path caches it at load time and warns if it changes
    // afterwards ("convolution gemm will use load-time value").
    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.opt.num_threads = threads;
    clk::time_point t0 = clk::now();
    if (net.load_param(argv[1]) != 0 || net.load_model(argv[2]) != 0) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }
    const double t_load = ms(clk::now() - t0);

    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f};

    double best_total = 1e18, best_resize = 0, best_mat = 0, best_fwd = 0;
    int checksum = 0;
    for (int r = 0; r < reps; r++) {
        double d_resize = 0, d_mat = 0, d_fwd = 0;
        clk::time_point tt0 = clk::now();
        for (int i = 0; i < n; i++) {
            clk::time_point a = clk::now();
            cv::Mat s;
            if (!use_ncnn_resize)
                cv::resize(raw[i], s, cv::Size(img_size, img_size), 0, 0, cv::INTER_AREA);
            clk::time_point b = clk::now();

            // Stride overload either way, so a crop that is a view into a larger image (what a
            // detector hands over) is read correctly rather than sheared.
            ncnn::Mat in = use_ncnn_resize
                ? ncnn::Mat::from_pixels_resize(raw[i].data, ncnn::Mat::PIXEL_BGR,
                                                raw[i].cols, raw[i].rows,
                                                static_cast<int>(raw[i].step),
                                                img_size, img_size)
                : ncnn::Mat::from_pixels(s.data, ncnn::Mat::PIXEL_BGR, s.cols, s.rows,
                                         static_cast<int>(s.step));
            in.substract_mean_normalize(mean_vals, norm_vals);
            clk::time_point c = clk::now();

            ncnn::Extractor ex = net.create_extractor();
            ex.input("in0", in);
            ncnn::Mat dists, margins;
            ex.extract("out0", dists);
            ex.extract("out1", margins);
            int best = 0;
            const float* dd = dists;
            for (int k = 1; k < dists.w; k++)
                if (dd[k] < dd[best]) best = k;
            checksum += best;                       // keep the work from being optimised away
            clk::time_point e = clk::now();

            d_resize += ms(b - a);
            d_mat += ms(c - b);
            d_fwd += ms(e - c);
        }
        const double total = ms(clk::now() - tt0);
        if (total < best_total) {
            best_total = total;
            best_resize = d_resize;
            best_mat = d_mat;
            best_fwd = d_fwd;
        }
    }

    printf("images        : %d   img_size=%d   threads=%d   reps=%d   resize=%s\n",
           n, img_size, threads, reps, resize_mode.c_str());
    printf("model load    : %8.2f ms  (once)\n", t_load);
    printf("resize        : %8.2f ms   (%.3f ms each)%s\n", best_resize, best_resize / n,
           use_ncnn_resize ? "  <- folded into the Mat build below" : "");
    printf("Mat%s+norm : %8.2f ms   (%.3f ms each)\n",
           use_ncnn_resize ? "+resize" : "      ", best_mat, best_mat / n);
    printf("forward       : %8.2f ms   (%.3f ms each)\n", best_fwd, best_fwd / n);
    printf("TOTAL         : %8.2f ms   (%.3f ms each, %.0f crops/s)\n",
           best_total, best_total / n, n * 1000.0 / best_total);
    printf("checksum      : %d\n", checksum);
    return 0;
}
