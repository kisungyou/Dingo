#include <dingo>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/SVD>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

struct CliOptions {
  bool quick = false;
  int samples = 0;
  std::vector<int> sizes;
  std::string csv_path;
};

struct ResultRow {
  std::string benchmark;
  int n = 0;
  int reps = 0;
  int samples = 0;
  double eigen_ms = 0.0;
  double dingo_ms = 0.0;
  double eigen_jitter_pct = 0.0;
  double dingo_jitter_pct = 0.0;
  double ratio = 0.0;
  double eigen_checksum = 0.0;
  double dingo_checksum = 0.0;
};

struct BenchCase {
  std::string name;
  std::function<ResultRow(int, const CliOptions&, std::mt19937_64&)> run;
};

struct TimingSummary {
  double median_ms = 0.0;
  double rel_std_pct = 0.0;
  double checksum = 0.0;
};

Eigen::MatrixXd make_random_matrix(int rows, int cols, std::mt19937_64& rng) {
  std::normal_distribution<double> dist(0.0, 1.0);
  Eigen::MatrixXd out(rows, cols);
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) {
      out(i, j) = dist(rng);
    }
  }
  return out;
}

Eigen::MatrixXd make_spd_matrix(int n, std::mt19937_64& rng) {
  Eigen::MatrixXd x = make_random_matrix(n, n, rng);
  Eigen::MatrixXd a = x.transpose() * x;
  a.diagonal().array() += static_cast<double>(n);
  return a;
}

std::vector<int> parse_sizes(const std::string& s) {
  std::vector<int> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (tok.empty()) {
      continue;
    }
    const int n = std::stoi(tok);
    if (n > 0) {
      out.push_back(n);
    }
  }
  return out;
}

CliOptions parse_cli(int argc, char** argv) {
  CliOptions opts;
  if (const char* env_quick = std::getenv("DINGO_BENCH_QUICK")) {
    opts.quick = std::string(env_quick) == "1";
  }

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--quick") {
      opts.quick = true;
    } else if (arg == "--sizes" && i + 1 < argc) {
      opts.sizes = parse_sizes(argv[++i]);
    } else if (arg == "--samples" && i + 1 < argc) {
      opts.samples = std::max(1, std::stoi(argv[++i]));
    } else if (arg == "--csv" && i + 1 < argc) {
      opts.csv_path = argv[++i];
    }
  }

  if (opts.sizes.empty()) {
    opts.sizes = opts.quick ? std::vector<int>{128, 256} : std::vector<int>{128, 256, 512};
  }
  if (opts.samples <= 0) {
    opts.samples = opts.quick ? 3 : 7;
  }
  return opts;
}

int reps_for(const std::string& op, int n, bool quick) {
  int reps = 1;
  if (op == "gemm") {
    reps = (n <= 128) ? 24 : (n <= 256 ? 12 : 4);
  } else if (op == "solve(qr)") {
    reps = (n <= 128) ? 16 : (n <= 256 ? 8 : 3);
  } else if (op == "lu") {
    reps = (n <= 128) ? 16 : (n <= 256 ? 10 : 4);
  } else if (op == "qr") {
    reps = (n <= 128) ? 10 : (n <= 256 ? 6 : 2);
  } else if (op == "svd") {
    reps = (n <= 128) ? 6 : (n <= 256 ? 3 : 1);
  } else if (op == "chol") {
    reps = (n <= 128) ? 18 : (n <= 256 ? 10 : 4);
  } else if (op == "eig_sym") {
    reps = (n <= 128) ? 6 : (n <= 256 ? 3 : 1);
  }
  int out = quick ? std::max(1, reps / 2) : reps;
  if (op == "svd" && out < 2) {
    out = 2;
  }
  return out;
}

double median_of(std::vector<double> x) {
  if (x.empty()) {
    return 0.0;
  }
  std::sort(x.begin(), x.end());
  const auto mid = x.size() / 2;
  if (x.size() % 2 == 1) {
    return x[mid];
  }
  return 0.5 * (x[mid - 1] + x[mid]);
}

double relative_std_pct(const std::vector<double>& x) {
  if (x.size() < 2) {
    return 0.0;
  }
  const double mean = std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(x.size());
  if (mean == 0.0) {
    return 0.0;
  }
  double sq = 0.0;
  for (double v : x) {
    const double d = v - mean;
    sq += d * d;
  }
  const double sd = std::sqrt(sq / static_cast<double>(x.size() - 1));
  return 100.0 * sd / mean;
}

template <typename Fn>
TimingSummary run_timed_stable(int reps, int samples, Fn&& fn) {
  volatile double sink = 0.0;
  std::vector<double> per_iter_ms;
  per_iter_ms.reserve(static_cast<std::size_t>(samples));

  // Single warmup for one-time initialization effects.
  sink += fn();

  for (int s = 0; s < samples; ++s) {
    const auto t0 = clock_type::now();
    for (int i = 0; i < reps; ++i) {
      sink += fn();
    }
    const auto t1 = clock_type::now();
    const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    per_iter_ms.push_back(total_ms / static_cast<double>(reps));
  }

  return {median_of(per_iter_ms), relative_std_pct(per_iter_ms), sink};
}

ResultRow run_gemm(int n, const CliOptions& opts, std::mt19937_64& rng) {
  const int reps = reps_for("gemm", n, opts.quick);
  const Eigen::MatrixXd a = make_random_matrix(n, n, rng);
  const Eigen::MatrixXd b = make_random_matrix(n, n, rng);
  const dingo::mat ad(a);
  const dingo::mat bd(b);

  const auto eigen = run_timed_stable(reps, opts.samples, [&]() {
    Eigen::MatrixXd c = a * b;
    return c.sum();
  });

  const auto dingo = run_timed_stable(reps, opts.samples, [&]() {
    dingo::mat c = ad * bd;
    return dingo::sum(c);
  });

  return {"gemm", n, reps, opts.samples, eigen.median_ms, dingo.median_ms, eigen.rel_std_pct,
          dingo.rel_std_pct, dingo.median_ms / eigen.median_ms, eigen.checksum, dingo.checksum};
}

ResultRow run_solve(int n, const CliOptions& opts, std::mt19937_64& rng) {
  const int reps = reps_for("solve(qr)", n, opts.quick);
  const Eigen::MatrixXd a = make_spd_matrix(n, rng);
  const Eigen::MatrixXd b = make_random_matrix(n, 8, rng);
  const dingo::mat ad(a);
  const dingo::mat bd(b);

  const auto eigen = run_timed_stable(reps, opts.samples, [&]() {
    Eigen::MatrixXd x = a.colPivHouseholderQr().solve(b);
    return x.sum();
  });

  const auto dingo = run_timed_stable(reps, opts.samples, [&]() {
    dingo::mat x = dingo::solve(ad, bd);
    return dingo::sum(x);
  });

  return {"solve(qr)", n, reps, opts.samples, eigen.median_ms, dingo.median_ms, eigen.rel_std_pct,
          dingo.rel_std_pct, dingo.median_ms / eigen.median_ms, eigen.checksum, dingo.checksum};
}

ResultRow run_lu(int n, const CliOptions& opts, std::mt19937_64& rng) {
  const int reps = reps_for("lu", n, opts.quick);
  const Eigen::MatrixXd a = make_random_matrix(n, n, rng);
  const dingo::mat ad(a);

  const auto eigen = run_timed_stable(reps, opts.samples, [&]() {
    Eigen::PartialPivLU<Eigen::MatrixXd> lu(a);
    Eigen::MatrixXd lu_m = lu.matrixLU();
    Eigen::MatrixXd l = Eigen::MatrixXd::Identity(n, n);
    l.triangularView<Eigen::StrictlyLower>() = lu_m.triangularView<Eigen::StrictlyLower>();
    Eigen::MatrixXd u = lu_m.triangularView<Eigen::Upper>();
    Eigen::MatrixXd p = lu.permutationP().toDenseMatrix().cast<double>();
    return l.sum() + u.sum() + p.sum();
  });

  const auto dingo = run_timed_stable(reps, opts.samples, [&]() {
    dingo::mat l;
    dingo::mat u;
    dingo::mat p;
    dingo::lu(l, u, p, ad);
    return dingo::sum(l) + dingo::sum(u) + dingo::sum(p);
  });

  return {"lu", n, reps, opts.samples, eigen.median_ms, dingo.median_ms, eigen.rel_std_pct,
          dingo.rel_std_pct, dingo.median_ms / eigen.median_ms, eigen.checksum, dingo.checksum};
}

ResultRow run_qr(int n, const CliOptions& opts, std::mt19937_64& rng) {
  const int reps = reps_for("qr", n, opts.quick);
  const Eigen::MatrixXd a = make_random_matrix(n, n, rng);
  const dingo::mat ad(a);

  const auto eigen = run_timed_stable(reps, opts.samples, [&]() {
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(a);
    Eigen::MatrixXd q = qr.householderQ() * Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd r = qr.matrixQR().triangularView<Eigen::Upper>();
    return q.sum() + r.sum();
  });

  const auto dingo = run_timed_stable(reps, opts.samples, [&]() {
    dingo::mat q;
    dingo::mat r;
    dingo::qr(q, r, ad);
    return dingo::sum(q) + dingo::sum(r);
  });

  return {"qr", n, reps, opts.samples, eigen.median_ms, dingo.median_ms, eigen.rel_std_pct,
          dingo.rel_std_pct, dingo.median_ms / eigen.median_ms, eigen.checksum, dingo.checksum};
}

ResultRow run_svd(int n, const CliOptions& opts, std::mt19937_64& rng) {
  const int reps = reps_for("svd", n, opts.quick);
  const Eigen::MatrixXd a = make_random_matrix(n, n, rng);
  const dingo::mat ad(a);

  const auto eigen = run_timed_stable(reps, opts.samples, [&]() {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(a, Eigen::ComputeThinU | Eigen::ComputeThinV);
    return svd.singularValues().sum();
  });

  const auto dingo = run_timed_stable(reps, opts.samples, [&]() {
    dingo::mat u;
    dingo::vec s;
    dingo::mat v;
    dingo::svd(u, s, v, ad);
    return s.eigen().sum();
  });

  return {"svd", n, reps, opts.samples, eigen.median_ms, dingo.median_ms, eigen.rel_std_pct,
          dingo.rel_std_pct, dingo.median_ms / eigen.median_ms, eigen.checksum, dingo.checksum};
}

ResultRow run_chol(int n, const CliOptions& opts, std::mt19937_64& rng) {
  const int reps = reps_for("chol", n, opts.quick);
  const Eigen::MatrixXd a = make_spd_matrix(n, rng);
  const dingo::mat ad(a);

  const auto eigen = run_timed_stable(reps, opts.samples, [&]() {
    Eigen::LLT<Eigen::MatrixXd> llt(a);
    return llt.matrixU().toDenseMatrix().sum();
  });

  const auto dingo = run_timed_stable(reps, opts.samples, [&]() {
    dingo::mat r;
    dingo::chol(r, ad);
    return dingo::sum(r);
  });

  return {"chol", n, reps, opts.samples, eigen.median_ms, dingo.median_ms, eigen.rel_std_pct,
          dingo.rel_std_pct, dingo.median_ms / eigen.median_ms, eigen.checksum, dingo.checksum};
}

ResultRow run_eig_sym(int n, const CliOptions& opts, std::mt19937_64& rng) {
  const int reps = reps_for("eig_sym", n, opts.quick);
  const Eigen::MatrixXd a = make_spd_matrix(n, rng);
  const dingo::mat ad(a);

  const auto eigen = run_timed_stable(reps, opts.samples, [&]() {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(a);
    return eig.eigenvalues().sum();
  });

  const auto dingo = run_timed_stable(reps, opts.samples, [&]() {
    dingo::vec eigval;
    dingo::mat eigvec;
    dingo::eig_sym(eigval, eigvec, ad);
    return eigval.eigen().sum();
  });

  return {"eig_sym", n, reps, opts.samples, eigen.median_ms, dingo.median_ms, eigen.rel_std_pct,
          dingo.rel_std_pct, dingo.median_ms / eigen.median_ms, eigen.checksum, dingo.checksum};
}

void print_header() {
  std::cout << std::left << std::setw(22) << "benchmark"
            << std::right << std::setw(8) << "N"
            << std::setw(8) << "reps"
            << std::setw(8) << "samp"
            << std::setw(14) << "eig_ms/op"
            << std::setw(14) << "din_ms/op"
            << std::setw(10) << "eig_cv%"
            << std::setw(10) << "din_cv%"
            << std::setw(10) << "ratio"
            << std::setw(16) << "checksum_gap" << '\n';
  std::cout << std::string(120, '-') << '\n';
}

void print_row(const ResultRow& r) {
  const double gap = std::abs(r.eigen_checksum - r.dingo_checksum);
  std::cout << std::left << std::setw(22) << r.benchmark
            << std::right << std::setw(8) << r.n
            << std::setw(8) << r.reps
            << std::setw(8) << r.samples
            << std::setw(14) << std::fixed << std::setprecision(3) << r.eigen_ms
            << std::setw(14) << std::fixed << std::setprecision(3) << r.dingo_ms
            << std::setw(10) << std::fixed << std::setprecision(2) << r.eigen_jitter_pct
            << std::setw(10) << std::fixed << std::setprecision(2) << r.dingo_jitter_pct
            << std::setw(10) << std::fixed << std::setprecision(3) << r.ratio
            << std::setw(16) << std::scientific << std::setprecision(3) << gap << '\n';
}

void write_csv(const std::string& path, const std::vector<ResultRow>& rows) {
  std::ofstream out(path);
  out << "benchmark,n,reps,samples,eigen_ms_per_op,dingo_ms_per_op,eigen_cv_pct,dingo_cv_pct,ratio,eigen_checksum,dingo_checksum,checksum_gap\n";
  for (const auto& r : rows) {
    const double gap = std::abs(r.eigen_checksum - r.dingo_checksum);
    out << r.benchmark << ','
        << r.n << ','
        << r.reps << ','
        << r.samples << ','
        << std::setprecision(12) << r.eigen_ms << ','
        << std::setprecision(12) << r.dingo_ms << ','
        << std::setprecision(12) << r.eigen_jitter_pct << ','
        << std::setprecision(12) << r.dingo_jitter_pct << ','
        << std::setprecision(12) << r.ratio << ','
        << std::setprecision(12) << r.eigen_checksum << ','
        << std::setprecision(12) << r.dingo_checksum << ','
        << std::setprecision(12) << gap << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  Eigen::setNbThreads(1);
  const CliOptions opts = parse_cli(argc, argv);

  std::cout << "Dingo vs Eigen runtime benchmark\n";
  std::cout << "mode: " << (opts.quick ? "quick" : "full") << "\n";
  std::cout << "Eigen threads: " << Eigen::nbThreads() << "\n";
  std::cout << "samples: " << opts.samples << "\n";
  std::cout << "sizes:";
  for (int n : opts.sizes) {
    std::cout << ' ' << n;
  }
  std::cout << "\n\n";

  std::vector<BenchCase> cases{
      {"gemm", run_gemm},       {"solve(qr)", run_solve}, {"lu", run_lu},      {"qr", run_qr},
      {"svd", run_svd},         {"chol", run_chol},       {"eig_sym", run_eig_sym},
  };

  std::mt19937_64 rng(20260222);
  std::vector<ResultRow> rows;
  rows.reserve(cases.size() * opts.sizes.size());

  for (const auto& c : cases) {
    for (int n : opts.sizes) {
      rows.push_back(c.run(n, opts, rng));
    }
  }

  print_header();
  for (const auto& r : rows) {
    print_row(r);
  }
  std::cout << "\nratio = dingo_ms_per_op / eigen_ms_per_op (lower is faster)\n";

  if (!opts.csv_path.empty()) {
    write_csv(opts.csv_path, rows);
    std::cout << "csv: " << opts.csv_path << '\n';
  }
  return 0;
}
