#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "dimreduce.hpp"
#include "rndgen.hpp"

#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace toolbox;

namespace {

class CerrSilencer {
    bool enabled_;
    std::streambuf* old_;
    std::ostringstream sink_;

public:
    explicit CerrSilencer(bool quiet) : enabled_(quiet), old_(nullptr) {
        if (enabled_) old_ = std::cerr.rdbuf(sink_.rdbuf());
    }

    ~CerrSilencer() {
        if (enabled_) std::cerr.rdbuf(old_);
    }
};

struct MetricBundle {
    NLDRMetricEuclid euclid;
    NLDRMetricPBC pbc;
    NLDRMetricSphere sphere;
    NLDRMetricDot dot;

    NLDRMetric* select(unsigned long dim, double period, double sphere_period, bool use_dot) {
        if (use_dot) {
            if (period != 0.0 || sphere_period != 0.0) {
                throw std::invalid_argument("dot metric cannot be combined with periodic metrics");
            }
            return &dot;
        }

        if (period == 0.0 && sphere_period == 0.0) return &euclid;
        if (sphere_period == 0.0) {
            pbc.periods.resize(dim);
            pbc.periods = period;
            return &pbc;
        }

        sphere.periods.resize(dim);
        sphere.periods = sphere_period;
        return &sphere;
    }
};

FMatrix<double> numpy_to_matrix(
    py::array_t<double, py::array::c_style | py::array::forcecast> array,
    const char* name
) {
    py::buffer_info info = array.request();
    if (info.ndim != 2) {
        throw std::invalid_argument(std::string(name) + " must be a two-dimensional array");
    }
    if (info.shape[0] <= 0 || info.shape[1] <= 0) {
        throw std::invalid_argument(std::string(name) + " must not be empty");
    }

    const auto rows = static_cast<unsigned long>(info.shape[0]);
    const auto cols = static_cast<unsigned long>(info.shape[1]);
    const double* data = static_cast<const double*>(info.ptr);

    FMatrix<double> out(rows, cols);
    for (unsigned long i = 0; i < rows; ++i) {
        for (unsigned long j = 0; j < cols; ++j) {
            out(i, j) = data[i * cols + j];
        }
    }
    return out;
}

std::valarray<double> numpy_to_weights(
    py::object obj,
    unsigned long expected,
    const char* name
) {
    if (obj.is_none()) return std::valarray<double>(0);

    py::array_t<double, py::array::c_style | py::array::forcecast> array =
        py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(obj);
    py::buffer_info info = array.request();
    if (info.ndim != 1) {
        throw std::invalid_argument(std::string(name) + " must be one-dimensional");
    }
    if (static_cast<unsigned long>(info.shape[0]) != expected) {
        throw std::invalid_argument(std::string(name) + " length does not match number of points");
    }

    const double* data = static_cast<const double*>(info.ptr);
    std::valarray<double> out(expected);
    for (unsigned long i = 0; i < expected; ++i) out[i] = data[i];
    return out;
}

py::array_t<double> matrix_to_numpy(const FMatrix<double>& matrix) {
    const auto rows = static_cast<py::ssize_t>(matrix.rows());
    const auto cols = static_cast<py::ssize_t>(matrix.cols());
    py::array_t<double> out({rows, cols});
    py::buffer_info info = out.request();
    double* data = static_cast<double*>(info.ptr);

    for (unsigned long i = 0; i < matrix.rows(); ++i) {
        for (unsigned long j = 0; j < matrix.cols(); ++j) {
            data[i * matrix.cols() + j] = matrix(i, j);
        }
    }
    return out;
}

py::array_t<double> valarray_to_numpy(const std::valarray<double>& values) {
    py::array_t<double> out({static_cast<py::ssize_t>(values.size())});
    py::buffer_info info = out.request();
    double* data = static_cast<double*>(info.ptr);
    for (unsigned long i = 0; i < values.size(); ++i) data[i] = values[i];
    return out;
}

py::array_t<unsigned long> indices_to_numpy(const std::vector<unsigned long>& values) {
    py::array_t<unsigned long> out({static_cast<py::ssize_t>(values.size())});
    py::buffer_info info = out.request();
    unsigned long* data = static_cast<unsigned long*>(info.ptr);
    for (unsigned long i = 0; i < values.size(); ++i) data[i] = values[i];
    return out;
}

std::vector<double> parse_number_sequence(py::object obj, const char* name) {
    std::vector<double> values;
    if (obj.is_none()) return values;

    if (py::isinstance<py::str>(obj)) {
        std::string raw = py::cast<std::string>(obj);
        if (raw == "identity" || raw.empty()) return values;
        std::stringstream ss(raw);
        std::string item;
        while (std::getline(ss, item, ',')) {
            if (!item.empty()) values.push_back(std::stod(item));
        }
        return values;
    }

    py::sequence seq = py::cast<py::sequence>(obj);
    for (py::handle item : seq) values.push_back(py::cast<double>(item));
    if (values.empty()) {
        throw std::invalid_argument(std::string(name) + " must not be an empty sequence");
    }
    return values;
}

void set_transfer_function(NLDRFunction& function, py::object obj, const char* name) {
    std::vector<double> parsed = parse_number_sequence(obj, name);
    std::valarray<double> pars(parsed.size());
    for (unsigned long i = 0; i < parsed.size(); ++i) pars[i] = parsed[i];

    if (parsed.empty()) {
        function.set_mode(NLDRIdentity, pars);
    } else if (parsed.size() == 2) {
#ifndef USE_BOOST
        throw std::invalid_argument(
            std::string(name) + " requested gamma mode, but this build was compiled without USE_BOOST"
        );
#else
        function.set_mode(NLDRGamma, pars);
#endif
    } else if (parsed.size() == 3) {
        function.set_mode(NLDRXSigmoid, pars);
    } else {
        throw std::invalid_argument(std::string(name) + " must be None, 'identity', (sigma,n), or (sigma,a,b)");
    }
}

bool parse_grid(py::object obj, double& width, unsigned long& coarse, unsigned long& fine) {
    if (obj.is_none()) return false;
    std::vector<double> parsed = parse_number_sequence(obj, "grid");
    if (parsed.size() != 3) {
        throw std::invalid_argument("grid must be a three-element sequence: (width, coarse_points, fine_points)");
    }
    width = parsed[0];
    coarse = static_cast<unsigned long>(parsed[1]);
    fine = static_cast<unsigned long>(parsed[2]);
    if (width <= 0.0 || coarse < 2 || fine < 2) {
        throw std::invalid_argument("grid width must be positive and grid sizes must be at least 2");
    }
    return true;
}

NLDRIterMin parse_minimizer(const std::string& mode) {
    if (mode == "conjgrad") return NLDRCGradient;
    if (mode == "simplex") return NLDRSimplex;
    if (mode == "anneal") return NLDRAnnealing;
    if (mode == "paratemp") return NLDRParatemp;
    if (mode == "nested") return NLDRNestSamp;
    throw std::invalid_argument("unknown minimizer mode: " + mode);
}

FMatrix<double> projection_low_points(NLDRProjection& projection) {
    std::valarray<std::valarray<double> > high;
    std::valarray<std::valarray<double> > low;
    projection.get_points(high, low);

    if (low.size() == 0) return FMatrix<double>(0, 0);
    FMatrix<double> out(low.size(), low[0].size());
    for (unsigned long i = 0; i < low.size(); ++i) {
        for (unsigned long j = 0; j < low[i].size(); ++j) {
            out(i, j) = low[i][j];
        }
    }
    return out;
}

void center_columns(FMatrix<double>& matrix) {
    for (unsigned long j = 0; j < matrix.cols(); ++j) {
        double mean = 0.0;
        for (unsigned long i = 0; i < matrix.rows(); ++i) mean += matrix(i, j);
        mean /= static_cast<double>(matrix.rows());
        for (unsigned long i = 0; i < matrix.rows(); ++i) matrix(i, j) -= mean;
    }
}

std::valarray<double> matrix_row(const FMatrix<double>& matrix, unsigned long row) {
    std::valarray<double> out(matrix.cols());
    for (unsigned long j = 0; j < matrix.cols(); ++j) {
        out[j] = const_cast<FMatrix<double>&>(matrix)(row, j);
    }
    return out;
}

void check_lowdim(unsigned long lowdim) {
    if (lowdim == 0) throw std::invalid_argument("lowdim must be at least 1");
}

py::dict py_mds(
    py::array_t<double, py::array::c_style | py::array::forcecast> points,
    unsigned long lowdim,
    double period,
    double sphere_period,
    bool dot,
    bool similarity,
    bool verbose,
    bool center,
    bool quiet
) {
    check_lowdim(lowdim);
    CerrSilencer silence(quiet);

    FMatrix<double> mpoints = numpy_to_matrix(points, "points");
    if (lowdim > mpoints.cols() && !similarity) {
        throw std::invalid_argument("lowdim cannot exceed input dimensionality");
    }
    if (similarity && mpoints.rows() != mpoints.cols()) {
        throw std::invalid_argument("similarity=True requires a square distance matrix");
    }

    MetricBundle metrics;
    NLDRMDSOptions opts;
    opts.lowdim = lowdim;
    opts.verbose = verbose;
    opts.metric = metrics.select(mpoints.cols(), period, sphere_period, dot);

    NLDRProjection projection;
    NLDRMDSReport report;
    if (similarity) NLDRMDS(mpoints, projection, opts, report, mpoints);
    else NLDRMDS(mpoints, projection, opts, report);

    FMatrix<double> low = projection_low_points(projection);
    if (center) center_columns(low);

    py::dict out;
    out["embedding"] = matrix_to_numpy(low);
    out["eigenvalues"] = valarray_to_numpy(report.deval);
    out["error"] = report.ld_error;
    if (report.ld_errors.size() > 0) out["per_point_errors"] = valarray_to_numpy(report.ld_errors);
    return out;
}

py::dict py_sketch_map(
    py::array_t<double, py::array::c_style | py::array::forcecast> points,
    unsigned long lowdim,
    py::object weights_obj,
    py::object init_obj,
    double period,
    double sphere_period,
    bool dot,
    py::object fun_hd,
    py::object fun_ld,
    unsigned long preopt_steps,
    py::object grid_obj,
    unsigned long global_steps,
    double imix,
    const std::string& minimizer,
    bool similarity,
    bool verbose,
    bool center,
    bool quiet
) {
    check_lowdim(lowdim);
    CerrSilencer silence(quiet);

    FMatrix<double> mpoints = numpy_to_matrix(points, "points");
    if (similarity && mpoints.rows() != mpoints.cols()) {
        throw std::invalid_argument("similarity=True requires a square distance matrix");
    }
    if (lowdim > mpoints.cols() && !similarity) {
        throw std::invalid_argument("lowdim cannot exceed input dimensionality");
    }

    MetricBundle metrics;
    NLDRMetric* metric = metrics.select(mpoints.cols(), period, sphere_period, dot);

    FMatrix<double> current;
    py::dict out;

    if (init_obj.is_none()) {
        NLDRMDSOptions mdsopts;
        NLDRMDSReport mdsreport;
        NLDRProjection mdsproj;
        mdsopts.lowdim = lowdim;
        mdsopts.verbose = false;
        mdsopts.metric = metric;
        if (similarity) NLDRMDS(mpoints, mdsproj, mdsopts, mdsreport, mpoints);
        else NLDRMDS(mpoints, mdsproj, mdsopts, mdsreport);
        current = projection_low_points(mdsproj);
        out["initial_error"] = mdsreport.ld_error;
    } else {
        current = numpy_to_matrix(
            py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(init_obj),
            "init"
        );
        if (current.rows() != mpoints.rows() || current.cols() != lowdim) {
            throw std::invalid_argument("init shape must be (n_points, lowdim)");
        }
    }

    double grid_width = 0.0;
    unsigned long grid_coarse = 0;
    unsigned long grid_fine = 0;
    bool use_grid = parse_grid(grid_obj, grid_width, grid_coarse, grid_fine);

    NLDRITERReport iterreport;
    if (preopt_steps > 0 || use_grid) {
        NLDRITEROptions iteropts;
        iteropts.lowdim = lowdim;
        iteropts.verbose = verbose;
        iteropts.metric = metric;
        iteropts.minmode = parse_minimizer(minimizer);
        iteropts.imix = imix;
        iteropts.weights = numpy_to_weights(weights_obj, mpoints.rows(), "weights");
        set_transfer_function(iteropts.tfunH, fun_hd, "fun_hd");
        set_transfer_function(iteropts.tfunL, fun_ld, "fun_ld");

        FMatrix<double> outd = similarity ? mpoints : FMatrix<double>(0, 0);

        if (preopt_steps > 0) {
            iteropts.global = false;
            iteropts.steps = preopt_steps;
            iteropts.ipoints = current;

            NLDRProjection preproj;
            NLDRITER(mpoints, preproj, iteropts, iterreport, outd);
            current = projection_low_points(preproj);
        }

        if (use_grid) {
            if (lowdim != 2) {
                throw std::invalid_argument("grid/global optimization is implemented only for lowdim=2");
            }
            iteropts.global = true;
            iteropts.steps = global_steps;
            iteropts.gridw = grid_width;
            iteropts.grid1 = grid_coarse;
            iteropts.grid2 = grid_fine;
            iteropts.ipoints = current;

            NLDRProjection globalproj;
            NLDRITER(mpoints, globalproj, iteropts, iterreport, outd);
            current = projection_low_points(globalproj);
        }
    }

    if (center) center_columns(current);
    out["embedding"] = matrix_to_numpy(current);
    if (iterreport.ld_errors.size() > 0) out["per_point_errors"] = valarray_to_numpy(iterreport.ld_errors);
    out["error"] = iterreport.ld_error;
    return out;
}

py::dict py_project(
    py::array_t<double, py::array::c_style | py::array::forcecast> samples,
    py::array_t<double, py::array::c_style | py::array::forcecast> landmarks_hd,
    py::array_t<double, py::array::c_style | py::array::forcecast> landmarks_ld,
    py::object weights_obj,
    double period,
    double sphere_period,
    bool dot,
    py::object fun_hd,
    py::object fun_ld,
    py::object grid_obj,
    unsigned long cg_steps,
    double grid_temperature,
    double path_lambda,
    bool similarity,
    bool quiet
) {
    CerrSilencer silence(quiet);

    FMatrix<double> sample_matrix = numpy_to_matrix(samples, "samples");
    FMatrix<double> hd = numpy_to_matrix(landmarks_hd, "landmarks_hd");
    FMatrix<double> ld = numpy_to_matrix(landmarks_ld, "landmarks_ld");
    if (hd.rows() != ld.rows()) {
        throw std::invalid_argument("landmarks_hd and landmarks_ld must contain the same number of rows");
    }
    if (!similarity && sample_matrix.cols() != hd.cols()) {
        throw std::invalid_argument("samples and landmarks_hd must have the same dimensionality");
    }
    if (similarity && sample_matrix.cols() != hd.rows()) {
        throw std::invalid_argument("similarity=True requires one distance per landmark in each sample row");
    }
    if (ld.cols() != 2) {
        throw std::invalid_argument("project currently uses the original 2D bicubic projector and requires landmarks_ld.shape[1] == 2");
    }

    double grid_width = 1.0;
    unsigned long grid_coarse = 21;
    unsigned long grid_fine = 201;
    parse_grid(grid_obj, grid_width, grid_coarse, grid_fine);

    MetricBundle metrics;
    NLDRMetric* metric = metrics.select(hd.cols(), period, sphere_period, dot);

    NLDROptions opts;
    opts.nopts.ometric = metric;
    opts.grid1 = grid_coarse;
    opts.grid2 = grid_fine;
    opts.gwidth = grid_width;
    opts.gtemp = grid_temperature;
    opts.cgsteps = cg_steps;
    set_transfer_function(opts.tfunH, fun_hd, "fun_hd");
    set_transfer_function(opts.tfunL, fun_ld, "fun_ld");

    std::valarray<double> weights = numpy_to_weights(weights_obj, hd.rows(), "weights");
    NLDRProjection projection;
    projection.set_options(opts);
    projection.set_points(hd, ld, weights);

    FMatrix<double> embedded(sample_matrix.rows(), ld.cols());
    std::valarray<double> errors(sample_matrix.rows());
    std::valarray<double> nearest(sample_matrix.rows());

    for (unsigned long i = 0; i < sample_matrix.rows(); ++i) {
        std::valarray<double> sample = matrix_row(sample_matrix, i);
        std::valarray<double> hp;
        std::valarray<double> lp(ld.cols());
        double mind = 0.0;
        double error = 0.0;

        if (path_lambda > 0.0) {
            lp = 0.0;
            double total_weight = 0.0;
            for (unsigned long j = 0; j < hd.rows(); ++j) {
                std::valarray<double> landmark = matrix_row(hd, j);
                double distance = similarity ? sample[j] : metric->dist(sample, landmark);
                double w = std::exp(-distance / path_lambda);
                total_weight += w;
                for (unsigned long k = 0; k < ld.cols(); ++k) {
                    lp[k] += ld(j, k) * w;
                }
                if (j == 0 || distance < mind) mind = distance;
            }
            for (unsigned long k = 0; k < ld.cols(); ++k) lp[k] /= total_weight;
            error = 0.0;
        } else {
            error = projection.project(sample, hp, lp, mind, similarity);
        }

        for (unsigned long k = 0; k < ld.cols(); ++k) embedded(i, k) = lp[k];
        errors[i] = error;
        nearest[i] = mind;
    }

    py::dict out;
    out["embedding"] = matrix_to_numpy(embedded);
    out["error"] = valarray_to_numpy(errors);
    out["nearest_distance"] = valarray_to_numpy(nearest);
    return out;
}

py::dict py_select_landmarks(
    py::array_t<double, py::array::c_style | py::array::forcecast> points,
    unsigned long n_landmarks,
    const std::string& mode,
    py::object input_weights_obj,
    double period,
    double sphere_period,
    bool dot,
    unsigned long seed,
    long first,
    bool unique,
    bool return_weights,
    double weight_gamma,
    bool quiet
) {
    CerrSilencer silence(quiet);
    FMatrix<double> hp = numpy_to_matrix(points, "points");
    const unsigned long n_points = hp.rows();
    const unsigned long dim = hp.cols();
    if (n_landmarks == 0 || n_landmarks > n_points) {
        throw std::invalid_argument("n_landmarks must be between 1 and number of input points");
    }

    std::valarray<double> input_weights = numpy_to_weights(input_weights_obj, n_points, "input_weights");
    if (input_weights.size() == 0) {
        input_weights.resize(n_points);
        input_weights = 1.0;
    }

    MetricBundle metrics;
    NLDRMetric* metric = metrics.select(dim, period, sphere_period, dot);
    FMatrix<double> landmarks(n_landmarks, dim);
    std::vector<unsigned long> selected(n_landmarks, 0);
    std::valarray<double> min_dist(n_points);
    min_dist = std::numeric_limits<double>::max();

    StdRndUniform rng(seed);
    rng();

    auto copy_landmark = [&](unsigned long out_index, unsigned long point_index) {
        selected[out_index] = point_index;
        for (unsigned long j = 0; j < dim; ++j) landmarks(out_index, j) = hp(point_index, j);
        for (unsigned long p = 0; p < n_points; ++p) {
            double distance = metric->dist(&landmarks(out_index, 0), &hp(p, 0), dim);
            if (distance < min_dist[p]) min_dist[p] = distance;
        }
    };

    auto already_selected = [&](unsigned long idx, unsigned long count) {
        for (unsigned long i = 0; i < count; ++i) {
            if (selected[i] == idx) return true;
        }
        return false;
    };

    if (mode == "stride") {
        unsigned long stride = n_points / n_landmarks;
        if (stride == 0) stride = 1;
        for (unsigned long i = 0; i < n_landmarks; ++i) {
            unsigned long idx = i * stride;
            if (idx >= n_points) idx = n_points - 1;
            copy_landmark(i, idx);
        }
    } else if (mode == "random") {
        for (unsigned long i = 0; i < n_landmarks; ++i) {
            unsigned long idx = 0;
            do {
                idx = static_cast<unsigned long>(rng() * n_points);
                if (idx >= n_points) idx = n_points - 1;
            } while (unique && already_selected(idx, i));
            copy_landmark(i, idx);
        }
    } else if (mode == "minmax") {
        unsigned long first_idx = first < 0 ? static_cast<unsigned long>(rng() * n_points) : static_cast<unsigned long>(first);
        if (first_idx >= n_points) throw std::invalid_argument("first index is out of range");
        copy_landmark(0, first_idx);

        for (unsigned long i = 1; i < n_landmarks; ++i) {
            double max_distance = -1.0;
            unsigned long max_index = 0;
            for (unsigned long p = 0; p < n_points; ++p) {
                if (min_dist[p] > max_distance) {
                    max_distance = min_dist[p];
                    max_index = p;
                }
            }
            copy_landmark(i, max_index);
        }
    } else {
        throw std::invalid_argument("mode must be 'minmax', 'random', or 'stride'");
    }

    py::dict out;
    out["landmarks"] = matrix_to_numpy(landmarks);
    out["indices"] = indices_to_numpy(selected);

    if (return_weights) {
        std::valarray<double> weights(n_landmarks);
        weights = 0.0;
        for (unsigned long p = 0; p < n_points; ++p) {
            double best = metric->dist(&landmarks(0, 0), &hp(p, 0), dim);
            unsigned long best_idx = 0;
            for (unsigned long l = 1; l < n_landmarks; ++l) {
                double distance = metric->dist(&landmarks(l, 0), &hp(p, 0), dim);
                if (distance < best) {
                    best = distance;
                    best_idx = l;
                }
            }
            weights[best_idx] += input_weights[p];
        }

        double total = 0.0;
        for (unsigned long i = 0; i < n_landmarks; ++i) {
            weights[i] = std::pow(weights[i], weight_gamma);
            total += weights[i];
        }
        if (total > 0.0) weights *= 1.0 / total;
        out["weights"] = valarray_to_numpy(weights);
    }

    return out;
}

py::array_t<double> py_pairwise_distances(
    py::array_t<double, py::array::c_style | py::array::forcecast> points,
    double period,
    double sphere_period,
    bool dot
) {
    FMatrix<double> hp = numpy_to_matrix(points, "points");
    MetricBundle metrics;
    NLDRMetric* metric = metrics.select(hp.cols(), period, sphere_period, dot);

    FMatrix<double> dist(hp.rows(), hp.rows());
    for (unsigned long i = 0; i < hp.rows(); ++i) {
        dist(i, i) = 0.0;
        for (unsigned long j = 0; j < i; ++j) {
            dist(i, j) = dist(j, i) = metric->dist(&hp(i, 0), &hp(j, 0), hp.cols());
        }
    }
    return matrix_to_numpy(dist);
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "pybind11 bindings for the sketch-map C++ core";
    m.attr("__version__") = "0.1.0";

    m.def(
        "mds",
        &py_mds,
        py::arg("points"),
        py::arg("lowdim") = 2,
        py::arg("period") = 0.0,
        py::arg("sphere_period") = 0.0,
        py::arg("dot") = false,
        py::arg("similarity") = false,
        py::arg("verbose") = false,
        py::arg("center") = false,
        py::arg("quiet") = true,
        "Run classical MDS using the C++ core."
    );

    m.def(
        "sketch_map",
        &py_sketch_map,
        py::arg("points"),
        py::arg("lowdim") = 2,
        py::arg("weights") = py::none(),
        py::arg("init") = py::none(),
        py::arg("period") = 0.0,
        py::arg("sphere_period") = 0.0,
        py::arg("dot") = false,
        py::arg("fun_hd") = py::none(),
        py::arg("fun_ld") = py::none(),
        py::arg("preopt_steps") = 100,
        py::arg("grid") = py::none(),
        py::arg("global_steps") = 0,
        py::arg("imix") = 0.0,
        py::arg("minimizer") = "conjgrad",
        py::arg("similarity") = false,
        py::arg("verbose") = false,
        py::arg("center") = false,
        py::arg("quiet") = true,
        "Run iterative sketch-map/MDS optimization using the C++ core."
    );

    m.def(
        "project",
        &py_project,
        py::arg("samples"),
        py::arg("landmarks_hd"),
        py::arg("landmarks_ld"),
        py::arg("weights") = py::none(),
        py::arg("period") = 0.0,
        py::arg("sphere_period") = 0.0,
        py::arg("dot") = false,
        py::arg("fun_hd") = py::none(),
        py::arg("fun_ld") = py::none(),
        py::arg("grid") = py::make_tuple(1.0, 21, 201),
        py::arg("cg_steps") = 0,
        py::arg("grid_temperature") = 0.0,
        py::arg("path_lambda") = -1.0,
        py::arg("similarity") = false,
        py::arg("quiet") = true,
        "Project new samples into an existing two-dimensional landmark map."
    );

    m.def(
        "select_landmarks",
        &py_select_landmarks,
        py::arg("points"),
        py::arg("n_landmarks"),
        py::arg("mode") = "minmax",
        py::arg("input_weights") = py::none(),
        py::arg("period") = 0.0,
        py::arg("sphere_period") = 0.0,
        py::arg("dot") = false,
        py::arg("seed") = 12345,
        py::arg("first") = -1,
        py::arg("unique") = false,
        py::arg("return_weights") = true,
        py::arg("weight_gamma") = 1.0,
        py::arg("quiet") = true,
        "Select landmarks using stride, random, or minmax selection."
    );

    m.def(
        "pairwise_distances",
        &py_pairwise_distances,
        py::arg("points"),
        py::arg("period") = 0.0,
        py::arg("sphere_period") = 0.0,
        py::arg("dot") = false,
        "Compute the pairwise distance matrix with the C++ metrics."
    );
}

