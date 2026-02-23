#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <dingo>

TEST_CASE("solve overloads and inverse family", "[linalg]") {
  const dingo::mat a{{4.0, 2.0}, {1.0, 3.0}};
  const dingo::mat b{{1.0}, {2.0}};
  const dingo::colvec bv{1.0, 2.0};

  const auto xm = dingo::solve(a, b);
  CHECK(xm(0, 0) == Catch::Approx(-0.1).margin(1e-6));
  CHECK(xm(1, 0) == Catch::Approx(0.7).margin(1e-6));

  const auto xv = dingo::solve(a, bv);
  CHECK(xv(0) == Catch::Approx(-0.1).margin(1e-6));
  CHECK(xv(1) == Catch::Approx(0.7).margin(1e-6));

  const auto ai = dingo::inv(a);
  const auto i = a * ai;
  CHECK(i(0, 0) == Catch::Approx(1.0).margin(1e-6));
  CHECK(i(1, 1) == Catch::Approx(1.0).margin(1e-6));

  const dingo::mat rank_def{{1.0, 2.0}, {2.0, 4.0}};
  const auto p = dingo::pinv(rank_def);
  const auto rec = rank_def * p * rank_def;
  CHECK(rec(0, 0) == Catch::Approx(rank_def(0, 0)).margin(1e-6));
  CHECK(rec(1, 1) == Catch::Approx(rank_def(1, 1)).margin(1e-6));
}

TEST_CASE("det rank cond", "[linalg]") {
  const dingo::mat a{{4.0, 2.0}, {1.0, 3.0}};
  CHECK(dingo::det(a) == Catch::Approx(10.0));
  CHECK(dingo::rank(a) == 2);
  CHECK(dingo::cond(a) > 1.0);

  const dingo::mat s{{1.0, 2.0}, {2.0, 4.0}};
  CHECK(dingo::rank(s) == 1);
  CHECK(std::isinf(dingo::cond(s)));
}

TEST_CASE("eigendecomposition variants", "[linalg]") {
  const dingo::mat sym{{2.0, 0.0}, {0.0, 3.0}};

  dingo::vec eval;
  dingo::mat evec;
  REQUIRE(dingo::eig_sym(eval, evec, sym));
  CHECK(eval.n_elem() == 2);
  CHECK(eval(0) == Catch::Approx(2.0).margin(1e-8));
  CHECK(eval(1) == Catch::Approx(3.0).margin(1e-8));

  const auto eval_only = dingo::eig_sym(sym);
  CHECK(eval_only.n_elem() == 2);
  CHECK(eval_only(0) == Catch::Approx(2.0).margin(1e-8));

  const dingo::mat nonsym{{0.0, -1.0}, {1.0, 0.0}};
  dingo::cx_vec geval;
  dingo::cx_mat gevec;
  REQUIRE(dingo::eig_gen(geval, gevec, nonsym));
  CHECK(geval.n_elem() == 2);
  CHECK(dingo::abs(geval.as_mat())(0, 0) == Catch::Approx(1.0).margin(1e-6));
}

TEST_CASE("svd qr chol lu log_det", "[linalg]") {
  const dingo::mat a{{2.0, 0.0}, {0.0, 3.0}};

  dingo::mat u;
  dingo::vec s;
  dingo::mat v;
  REQUIRE(dingo::svd(u, s, v, a));
  CHECK(s.n_elem() == 2);
  CHECK(s(0) == Catch::Approx(3.0).margin(1e-8));
  CHECK(s(1) == Catch::Approx(2.0).margin(1e-8));
  const auto s_only = dingo::svd(a);
  CHECK(s_only.n_elem() == 2);

  dingo::mat q;
  dingo::mat r;
  REQUIRE(dingo::qr(q, r, a));
  const auto qr = q * r;
  CHECK(qr(0, 0) == Catch::Approx(a(0, 0)).margin(1e-6));
  CHECK(qr(1, 1) == Catch::Approx(a(1, 1)).margin(1e-6));

  const dingo::mat spd{{4.0, 1.0}, {1.0, 3.0}};
  dingo::mat cu;
  REQUIRE(dingo::chol(cu, spd));
  const auto recon_u = dingo::trans(cu) * cu;
  CHECK(recon_u(0, 0) == Catch::Approx(spd(0, 0)).margin(1e-6));

  dingo::mat cl;
  REQUIRE(dingo::chol(cl, spd, "lower"));
  const auto recon_l = cl * dingo::trans(cl);
  CHECK(recon_l(1, 1) == Catch::Approx(spd(1, 1)).margin(1e-6));

  dingo::mat l;
  dingo::mat uu;
  dingo::mat p;
  REQUIRE(dingo::lu(l, uu, p, spd));
  CHECK((p * spd)(0, 0) == Catch::Approx((l * uu)(0, 0)).margin(1e-6));

  double logv = 0.0;
  double sign = 0.0;
  REQUIRE(dingo::log_det(logv, sign, spd));
  CHECK(sign == Catch::Approx(1.0));
  CHECK(logv == Catch::Approx(std::log(dingo::det(spd))).margin(1e-8));
}

TEST_CASE("failure-mode behavior", "[linalg]") {
  const dingo::mat non_spd{{1.0, 2.0}, {2.0, -1.0}};
  dingo::mat out;
  CHECK_FALSE(dingo::chol(out, non_spd));

  const dingo::mat singular{{1.0, 2.0}, {2.0, 4.0}};
  double logv = 0.0;
  double sign = 1.0;
  CHECK_FALSE(dingo::log_det(logv, sign, singular));
  CHECK(sign == Catch::Approx(0.0));
  CHECK(std::isinf(logv));
}

TEST_CASE("kron null orth", "[linalg]") {
  const dingo::mat a{{1.0, 2.0}, {3.0, 4.0}};
  const dingo::mat b{{0.0, 5.0}, {6.0, 7.0}};
  const auto k = dingo::kron(a, b);
  CHECK(k.n_rows() == 4);
  CHECK(k.n_cols() == 4);
  CHECK(k(0, 1) == Catch::Approx(5.0));
  CHECK(k(3, 2) == Catch::Approx(24.0));

  const dingo::mat x{{1.0, 1.0}, {1.0, 1.0}, {0.0, 0.0}};
  const auto n = dingo::null(x);
  CHECK(n.n_rows() == 2);
  CHECK(n.n_cols() == 1);
  const auto xn = x * n;
  CHECK(std::abs(xn(0, 0)) < 1e-8);
  CHECK(std::abs(xn(1, 0)) < 1e-8);

  const dingo::mat y{{1.0, 1.0}, {0.0, 1.0}, {0.0, 0.0}};
  const auto q = dingo::orth(y);
  CHECK(q.n_rows() == 3);
  CHECK(q.n_cols() == 2);
  const auto qtq = dingo::trans(q) * q;
  CHECK(qtq(0, 0) == Catch::Approx(1.0).margin(1e-8));
  CHECK(qtq(1, 1) == Catch::Approx(1.0).margin(1e-8));
  CHECK(qtq(0, 1) == Catch::Approx(0.0).margin(1e-8));
}
