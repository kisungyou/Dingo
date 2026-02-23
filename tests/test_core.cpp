#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <dingo>

#include <Eigen/Core>

#include <complex>
#include <string>

TEST_CASE("Mat constructors, access, and mutation", "[core]") {
  dingo::mat empty;
  CHECK(empty.is_empty());

  dingo::mat a(2, 3, dingo::fill::ones);
  CHECK(a.n_rows() == 2);
  CHECK(a.n_cols() == 3);
  CHECK(a.n_elem() == 6);
  CHECK(a(0, 0) == Catch::Approx(1.0));

  a.zeros();
  CHECK(a(1, 2) == Catch::Approx(0.0));
  a.ones();
  CHECK(a(1, 2) == Catch::Approx(1.0));
  CHECK(a(1) == Catch::Approx(1.0));  // column-major linear index
  a.randu();
  a.randn();

  a(0, 1) = 4.0;
  CHECK(a.at(0, 1) == Catch::Approx(4.0));

  a.resize(3, 2);
  CHECK(a.n_rows() == 3);
  CHECK(a.n_cols() == 2);
}

TEST_CASE("Mat row/col/submat helpers", "[core]") {
  dingo::mat a{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

  const auto r1 = a.row(1);
  const auto c2 = a.col(2);
  CHECK(r1.n_rows() == 1);
  CHECK(r1.n_cols() == 3);
  CHECK(c2.n_rows() == 3);
  CHECK(c2.n_cols() == 1);
  CHECK(r1(0, 1) == Catch::Approx(5.0));
  CHECK(c2(2, 0) == Catch::Approx(9.0));

  a.set_row(0, dingo::mat{{10.0, 11.0, 12.0}});
  a.set_col(1, dingo::mat{{20.0}, {21.0}, {22.0}});
  CHECK(a(0, 0) == Catch::Approx(10.0));
  CHECK(a(2, 1) == Catch::Approx(22.0));

  const auto b = a.submat(1, 0, 2, 1);
  CHECK(b.n_rows() == 2);
  CHECK(b.n_cols() == 2);
  CHECK(b(0, 0) == Catch::Approx(4.0));

  dingo::mat z(3, 3, dingo::fill::zeros);
  z.set_submat(1, 1, dingo::mat{{1.0, 2.0}, {3.0, 4.0}});
  CHECK(z(1, 1) == Catch::Approx(1.0));
  CHECK(z(2, 2) == Catch::Approx(4.0));
}

TEST_CASE("Mat arithmetic and transpose variants", "[core]") {
  dingo::mat a{{1.0, 2.0}, {3.0, 4.0}};
  dingo::mat b{{5.0, 6.0}, {7.0, 8.0}};

  CHECK((a + b)(0, 0) == Catch::Approx(6.0));
  CHECK((b - a)(1, 1) == Catch::Approx(4.0));
  CHECK((a % b)(0, 1) == Catch::Approx(12.0));
  CHECK((a * b)(1, 1) == Catch::Approx(50.0));
  CHECK((b / a)(0, 0) == Catch::Approx(5.0));
  CHECK((2.0 * a)(1, 0) == Catch::Approx(6.0));
  CHECK((a * 2.0)(1, 0) == Catch::Approx(6.0));
  CHECK((b / 2.0)(1, 1) == Catch::Approx(4.0));
  CHECK((-a)(0, 1) == Catch::Approx(-2.0));

  const auto at = a.t();
  CHECK(at(0, 1) == Catch::Approx(3.0));
  const auto ast = a.st();
  CHECK(ast(0, 1) == Catch::Approx(3.0));
  CHECK(dingo::trans(a)(1, 0) == Catch::Approx(2.0));
  CHECK(dingo::strans(a)(1, 0) == Catch::Approx(2.0));
}

TEST_CASE("Factory, reshape, vectorise, joins, and grids", "[core]") {
  const auto z = dingo::zeros(2, 3);
  const auto o = dingo::ones(2, 3);
  CHECK(z.n_elem() == 6);
  CHECK(o(1, 2) == Catch::Approx(1.0));

  const auto zv = dingo::zeros(3);
  const auto ov = dingo::ones(3);
  CHECK(zv.n_elem() == 3);
  CHECK(ov(2) == Catch::Approx(1.0));

  const auto ru = dingo::randu(2, 2);
  const auto rn = dingo::randn(2, 2);
  CHECK(ru.n_rows() == 2);
  CHECK(rn.n_cols() == 2);
  const auto ruv = dingo::randu(5);
  const auto rnv = dingo::randn(5);
  CHECK(ruv.n_elem() == 5);
  CHECK(rnv.n_elem() == 5);

  const auto i = dingo::eye(3);
  CHECK(i(0, 0) == Catch::Approx(1.0));
  CHECK(i(0, 1) == Catch::Approx(0.0));
  CHECK(dingo::eye(2, 3).n_cols() == 3);

  const auto ls = dingo::linspace(0.0, 1.0, 5);
  CHECK(ls(0) == Catch::Approx(0.0));
  CHECK(ls(4) == Catch::Approx(1.0));
  const auto rg = dingo::regspace(1.0, 2.0, 7.0);
  CHECK(rg.n_elem() == 4);
  CHECK(rg(3) == Catch::Approx(7.0));

  const dingo::mat a{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  const auto v = dingo::vectorise(a);
  CHECK(v.n_elem() == 6);
  CHECK(v(1) == Catch::Approx(4.0));
  const auto r = dingo::reshape(v.as_mat(), 3, 2);
  CHECK(r.n_rows() == 3);
  CHECK(r(1, 0) == Catch::Approx(4.0));

  const dingo::mat l{{1.0}, {2.0}};
  const dingo::mat rj{{3.0}, {4.0}};
  CHECK(dingo::join_rows(l, rj).n_cols() == 2);
  CHECK(dingo::join_cols(l.t(), rj.t()).n_rows() == 2);
}

TEST_CASE("Reductions, norms, and structure helpers", "[core]") {
  const dingo::mat a{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  CHECK(dingo::sum(a) == Catch::Approx(21.0));
  CHECK(dingo::accu(a) == Catch::Approx(21.0));
  CHECK(dingo::mean(a) == Catch::Approx(3.5));
  CHECK(dingo::min(a) == Catch::Approx(1.0));
  CHECK(dingo::max(a) == Catch::Approx(6.0));
  CHECK(dingo::trace(dingo::mat{{2.0, 1.0}, {3.0, 4.0}}) == Catch::Approx(6.0));
  CHECK(dingo::var(a) == Catch::Approx(3.5));
  CHECK(dingo::stddev(a) == Catch::Approx(std::sqrt(3.5)));

  const auto s0 = dingo::sum(a, 0);
  const auto s1 = dingo::sum(a, 1);
  CHECK(s0.n_rows() == 1);
  CHECK(s0.n_cols() == 3);
  CHECK(s0(0, 0) == Catch::Approx(5.0));
  CHECK(s1.n_rows() == 2);
  CHECK(s1(1, 0) == Catch::Approx(15.0));

  const auto m0 = dingo::mean(a, 0);
  const auto m1 = dingo::mean(a, 1);
  CHECK(m0(0, 1) == Catch::Approx(3.5));
  CHECK(m1(0, 0) == Catch::Approx(2.0));

  CHECK(dingo::norm(a, 2) == Catch::Approx(a.eigen().norm()));
  CHECK(dingo::norm(a, 1) == Catch::Approx(a.eigen().lpNorm<1>()));
  CHECK(dingo::norm(a, 0) == Catch::Approx(6.0));
  CHECK_THROWS_AS(dingo::norm(a, 3), std::invalid_argument);

  const auto dv = dingo::diagvec(dingo::mat{{1.0, 2.0}, {3.0, 4.0}});
  CHECK(dv.n_elem() == 2);
  CHECK(dv(1) == Catch::Approx(4.0));
  CHECK(dingo::diagmat(dv)(1, 1) == Catch::Approx(4.0));
  CHECK(dingo::diagmat(dingo::mat{{1.0, 2.0}, {3.0, 4.0}})(0, 1) == Catch::Approx(0.0));

  CHECK(dingo::tril(dingo::mat{{1.0, 2.0}, {3.0, 4.0}})(0, 1) == Catch::Approx(0.0));
  CHECK(dingo::triu(dingo::mat{{1.0, 2.0}, {3.0, 4.0}})(1, 0) == Catch::Approx(0.0));
  CHECK(dingo::tril(dingo::mat{{1.0, 2.0}, {3.0, 4.0}}, 1)(0, 1) == Catch::Approx(2.0));
  CHECK(dingo::triu(dingo::mat{{1.0, 2.0}, {3.0, 4.0}}, -1)(1, 0) == Catch::Approx(3.0));
}

TEST_CASE("Elementwise math helpers", "[core]") {
  const dingo::mat x{{0.0, 1.0}};
  CHECK(dingo::sin(x)(0, 0) == Catch::Approx(0.0));
  CHECK(dingo::cos(x)(0, 0) == Catch::Approx(1.0));
  CHECK(dingo::exp(x)(0, 0) == Catch::Approx(1.0));
  CHECK(dingo::log(dingo::mat{{1.0}})(0, 0) == Catch::Approx(0.0));
  CHECK(dingo::sqrt(dingo::mat{{4.0}})(0, 0) == Catch::Approx(2.0));

  const dingo::cx_mat c{{{3.0, 4.0}}};
  const auto a = dingo::abs(c);
  CHECK(a(0, 0) == Catch::Approx(5.0));
}

TEST_CASE("Col/Row vector helpers", "[core]") {
  dingo::colvec c{1.0, 2.0, 3.0};
  dingo::rowvec r{4.0, 5.0, 6.0};

  CHECK(c.n_rows() == 3);
  CHECK(r.n_cols() == 3);
  CHECK(c(1) == Catch::Approx(2.0));
  CHECK(r(2) == Catch::Approx(6.0));

  c.ones();
  r.zeros();
  CHECK(c(2) == Catch::Approx(1.0));
  CHECK(r(0) == Catch::Approx(0.0));

  c.randu();
  r.randn();
  CHECK(c.n_elem() == 3);
  CHECK(r.n_elem() == 3);

  CHECK(c.as_mat().n_cols() == 1);
  CHECK(r.as_mat().n_rows() == 1);

  CHECK(dingo::sum(c) == Catch::Approx(c.eigen().sum()));
  CHECK(dingo::sum(r) == Catch::Approx(r.eigen().sum()));
}

TEST_CASE("Cube and Cell methods", "[core]") {
  dingo::cube a(2, 2, 2, dingo::fill::zeros);
  dingo::cube b(2, 2, 1, dingo::fill::ones);
  a(1, 1, 1) = 7.0;
  CHECK(a(1, 1, 1) == Catch::Approx(7.0));
  CHECK(a.slice(1)(1, 1) == Catch::Approx(7.0));

  a.ones();
  CHECK(a(0, 0, 0) == Catch::Approx(1.0));
  a.zeros();
  CHECK(a(0, 0, 0) == Catch::Approx(0.0));
  a.randu();
  a.randn();

  const auto j = dingo::join_slices(a, b);
  CHECK(j.n_slices() == 3);
  CHECK(j.n_rows() == 2);
  CHECK(j.n_elem() == 12);

  a.resize(1, 3, 2);
  CHECK(a.n_rows() == 1);
  CHECK(a.n_cols() == 3);
  CHECK(a.n_slices() == 2);

  dingo::cell<std::string> empty_cells;
  CHECK(empty_cells.is_empty());
  dingo::cell<std::string> cells(2, 2);
  cells(0, 0) = "a";
  cells(1, 1) = "b";
  CHECK(cells.n_elem() == 4);
  CHECK(cells(1, 1) == "b");
  cells.resize(1, 3);
  CHECK(cells.n_cols() == 3);
  CHECK(cells.n_slices() == 1);
  cells.fill("x");
  CHECK(cells(0, 2) == "x");
}

TEST_CASE("Span indexing and broadcasting", "[core]") {
  const dingo::mat a{{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}, {9.0, 10.0, 11.0, 12.0}};
  const auto b = a.submat(dingo::range(1, 2), dingo::range(1, 3));
  CHECK(b.n_rows() == 2);
  CHECK(b.n_cols() == 3);
  CHECK(b(0, 0) == Catch::Approx(6.0));
  CHECK(b(1, 2) == Catch::Approx(12.0));

  const auto c = a.submat(dingo::all_idx, dingo::range(2, 3));
  CHECK(c.n_rows() == 3);
  CHECK(c.n_cols() == 2);
  CHECK(c(2, 1) == Catch::Approx(12.0));

  const auto d = a.submat(dingo::range(0, 1), dingo::all_idx);
  CHECK(d.n_rows() == 2);
  CHECK(d.n_cols() == 4);
  CHECK(d(1, 3) == Catch::Approx(8.0));

  const auto e = a.submat(dingo::all_idx, dingo::all_idx);
  CHECK(e.n_rows() == a.n_rows());
  CHECK(e.n_cols() == a.n_cols());

  dingo::mat w = a;
  const auto ws = w(dingo::range(1, 2), dingo::range(1, 2)).eval();
  CHECK(ws(0, 0) == Catch::Approx(6.0));
  w(dingo::range(1, 2), dingo::range(1, 2)) = dingo::mat{{100.0, 101.0}, {102.0, 103.0}};
  CHECK(w(1, 1) == Catch::Approx(100.0));
  CHECK(w(2, 2) == Catch::Approx(103.0));
  w(dingo::all_idx, dingo::range(0, 0)) = 9.0;
  CHECK(w(0, 0) == Catch::Approx(9.0));
  CHECK(w(2, 0) == Catch::Approx(9.0));

  const dingo::mat m{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  const dingo::colvec cv{10.0, 20.0};
  const dingo::rowvec rv{100.0, 200.0, 300.0};
  CHECK((m + cv)(1, 2) == Catch::Approx(26.0));
  CHECK((m + rv)(0, 1) == Catch::Approx(202.0));
  CHECK((m - cv)(1, 0) == Catch::Approx(-16.0));
  CHECK((m - rv)(1, 2) == Catch::Approx(-294.0));
  CHECK((m % cv)(1, 1) == Catch::Approx(100.0));
  CHECK((m % rv)(0, 2) == Catch::Approx(900.0));
  CHECK((m / cv)(1, 0) == Catch::Approx(0.2));
  CHECK((m / rv)(0, 0) == Catch::Approx(0.01));
}

TEST_CASE("Logical mask indexing and assignment", "[core]") {
  dingo::mat a{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  const dingo::mat mask{{0.0, 1.0, 0.0}, {1.0, 0.0, 1.0}};

  const auto picked = static_cast<const dingo::mat&>(a)(mask);
  CHECK(picked.n_rows() == 3);
  CHECK(picked.n_cols() == 1);
  CHECK(picked(0, 0) == Catch::Approx(4.0));
  CHECK(picked(1, 0) == Catch::Approx(2.0));
  CHECK(picked(2, 0) == Catch::Approx(6.0));

  a(mask) = -1.0;
  CHECK(a(0, 1) == Catch::Approx(-1.0));
  CHECK(a(1, 0) == Catch::Approx(-1.0));
  CHECK(a(1, 2) == Catch::Approx(-1.0));

  dingo::mat b{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  b(mask) = dingo::mat{{10.0}, {20.0}, {30.0}};
  CHECK(b(1, 0) == Catch::Approx(10.0));
  CHECK(b(0, 1) == Catch::Approx(20.0));
  CHECK(b(1, 2) == Catch::Approx(30.0));
}

TEST_CASE("Dim-wise reductions and utilities", "[core]") {
  const dingo::mat a{{1.0, 0.0, 3.0}, {4.0, 5.0, 0.0}};

  CHECK(dingo::min(a, 0)(0, 0) == Catch::Approx(1.0));
  CHECK(dingo::min(a, 1)(1, 0) == Catch::Approx(0.0));
  CHECK(dingo::max(a, 0)(0, 1) == Catch::Approx(5.0));
  CHECK(dingo::max(a, 1)(0, 0) == Catch::Approx(3.0));

  const auto v0 = dingo::var(a, 0);
  const auto v1 = dingo::var(a, 1);
  CHECK(v0(0, 0) == Catch::Approx(4.5));
  CHECK(v1(0, 0) == Catch::Approx(7.0 / 3.0));

  const auto s0 = dingo::stddev(a, 0);
  const auto s1 = dingo::stddev(a, 1);
  CHECK(s0(0, 0) == Catch::Approx(std::sqrt(4.5)));
  CHECK(s1(1, 0) == Catch::Approx(std::sqrt(7.0)));

  const auto any0 = dingo::any(a, 0);
  const auto any1 = dingo::any(a, 1);
  const auto all0 = dingo::all(a, 0);
  const auto all1 = dingo::all(a, 1);
  CHECK(any0(0, 1) == Catch::Approx(1.0));
  CHECK(any1(1, 0) == Catch::Approx(1.0));
  CHECK(all0(0, 1) == Catch::Approx(0.0));
  CHECK(all1(0, 0) == Catch::Approx(0.0));

  const auto ud = dingo::flipud(a);
  const auto lr = dingo::fliplr(a);
  CHECK(ud(0, 0) == Catch::Approx(4.0));
  CHECK(lr(0, 0) == Catch::Approx(3.0));

  const auto rep = dingo::repmat(dingo::mat{{1.0, 2.0}}, 2, 3);
  CHECK(rep.n_rows() == 2);
  CHECK(rep.n_cols() == 6);
  CHECK(rep(1, 4) == Catch::Approx(1.0));
}

TEST_CASE("Sorting and linear indexing helpers", "[core]") {
  const dingo::colvec c{3.0, 1.0, 2.0, 1.0};
  const dingo::rowvec r{4.0, 2.0, 3.0};
  const auto sc = dingo::sort(c);
  const auto sr = dingo::sort(r);
  CHECK(sc(0) == Catch::Approx(1.0));
  CHECK(sc(3) == Catch::Approx(3.0));
  CHECK(sr(0) == Catch::Approx(2.0));
  CHECK(sr(2) == Catch::Approx(4.0));

  const auto uq = dingo::unique(c);
  CHECK(uq.n_elem() == 3);
  CHECK(uq(1) == Catch::Approx(2.0));

  const dingo::mat a{{0.0, 5.0}, {6.0, 0.0}, {7.0, 8.0}};
  const auto idx = dingo::find(a);
  CHECK(idx.n_elem() == 4);
  CHECK(idx(0) == 1);
  CHECK(idx(3) == 5);
  const auto vals = dingo::elem(a, idx);
  CHECK(vals(0) == Catch::Approx(6.0));
  CHECK(vals(3) == Catch::Approx(8.0));
}

TEST_CASE("Complex and cube utilities", "[core]") {
  const dingo::cx_mat c{{{1.0, 2.0}, {3.0, -4.0}}};
  const auto cc = dingo::conj(c);
  CHECK(cc(0, 0).imag() == Catch::Approx(-2.0));

  const auto re = dingo::real(c);
  const auto im = dingo::imag(c);
  CHECK(re(0, 1) == Catch::Approx(3.0));
  CHECK(im(0, 1) == Catch::Approx(-4.0));

  const auto ang = dingo::angle(c);
  CHECK(ang(0, 0) == Catch::Approx(std::atan2(2.0, 1.0)));
  const auto a2 = dingo::abs2(c);
  CHECK(a2(0, 1) == Catch::Approx(25.0));

  dingo::cube x(2, 3, 1, dingo::fill::zeros);
  x(1, 2, 0) = 9.0;
  const auto sq = dingo::squeeze(x);
  CHECK(sq.n_rows() == 2);
  CHECK(sq.n_cols() == 3);
  CHECK(sq(1, 2) == Catch::Approx(9.0));

  dingo::cube y(2, 3, 4, dingo::fill::zeros);
  y(1, 2, 3) = 11.0;
  const auto p = dingo::permute(y, 3, 2, 1);
  CHECK(p.n_rows() == 4);
  CHECK(p.n_cols() == 3);
  CHECK(p.n_slices() == 2);
  CHECK(p(3, 2, 1) == Catch::Approx(11.0));
}

TEST_CASE("Eigen interop helpers", "[core]") {
  Eigen::MatrixXd m(2, 2);
  m << 1.0, 2.0, 3.0, 4.0;
  auto dm = dingo::from_eigen_mat(m);
  CHECK(dm(1, 1) == Catch::Approx(4.0));

  Eigen::VectorXd v(3);
  v << 5.0, 6.0, 7.0;
  auto dc = dingo::from_eigen_col(v);
  CHECK(dc(2) == Catch::Approx(7.0));

  Eigen::RowVectorXd rv(3);
  rv << 8.0, 9.0, 10.0;
  auto dr = dingo::from_eigen_row(rv);
  CHECK(dr(1) == Catch::Approx(9.0));

  dingo::mat a{{1.0, 2.0}, {3.0, 4.0}};
  auto& ae = dingo::as_eigen(a);
  ae(0, 0) = 42.0;
  CHECK(a(0, 0) == Catch::Approx(42.0));
}

TEST_CASE("Indexing, broadcasting, and dim-wise logical reductions", "[core]") {
  const dingo::mat a{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
  const auto s = a.submat(dingo::range(1, 2), dingo::range(0, 1));
  CHECK(s.n_rows() == 2);
  CHECK(s.n_cols() == 2);
  CHECK(s(1, 1) == Catch::Approx(8.0));

  const auto all_rows = a.submat(dingo::all_t{}, dingo::range(1, 2));
  CHECK(all_rows.n_rows() == 3);
  CHECK(all_rows.n_cols() == 2);
  CHECK(all_rows(2, 1) == Catch::Approx(9.0));

  const auto all_cols = a.submat(dingo::range(0, 1), dingo::all_t{});
  CHECK(all_cols.n_rows() == 2);
  CHECK(all_cols.n_cols() == 3);
  CHECK(all_cols(1, 2) == Catch::Approx(6.0));

  const dingo::colvec c{10.0, 20.0, 30.0};
  const dingo::rowvec r{1.0, 2.0, 3.0};
  const auto plus_c = a + c;
  const auto plus_r = a + r;
  CHECK(plus_c(2, 0) == Catch::Approx(37.0));
  CHECK(plus_r(0, 2) == Catch::Approx(6.0));
  CHECK((a - c)(1, 1) == Catch::Approx(-15.0));
  CHECK((a - r)(1, 2) == Catch::Approx(3.0));
  CHECK((a % c)(2, 1) == Catch::Approx(240.0));
  CHECK((a % r)(1, 2) == Catch::Approx(18.0));
  CHECK((a / c)(2, 0) == Catch::Approx(7.0 / 30.0));
  CHECK((a / r)(1, 2) == Catch::Approx(2.0));

  const dingo::mat b{{1.0, 0.0, 2.0}, {0.0, 0.0, 3.0}};
  CHECK(dingo::any(b, 0)(0, 1) == Catch::Approx(0.0));
  CHECK(dingo::any(b, 1)(1, 0) == Catch::Approx(1.0));
  CHECK(dingo::all(b, 0)(0, 0) == Catch::Approx(0.0));
  CHECK(dingo::all(dingo::mat{{1.0, 2.0}, {3.0, 4.0}}, 1)(1, 0) == Catch::Approx(1.0));
}

TEST_CASE("Extended matrix utilities and complex helpers", "[core]") {
  const dingo::mat a{{1.0, 2.0}, {3.0, 4.0}};
  CHECK(dingo::flipud(a)(0, 0) == Catch::Approx(3.0));
  CHECK(dingo::fliplr(a)(0, 0) == Catch::Approx(2.0));

  const auto rp = dingo::repmat(dingo::mat{{1.0, 2.0}}, 2, 3);
  CHECK(rp.n_rows() == 2);
  CHECK(rp.n_cols() == 6);
  CHECK(rp(1, 4) == Catch::Approx(1.0));

  dingo::colvec u{3.0, 1.0, 3.0, 2.0};
  const auto us = dingo::sort(u);
  CHECK(us(0) == Catch::Approx(1.0));
  CHECK(us(3) == Catch::Approx(3.0));
  const auto uq = dingo::unique(u);
  CHECK(uq.n_elem() == 3);
  CHECK(uq(0) == Catch::Approx(1.0));
  CHECK(uq(2) == Catch::Approx(3.0));

  const dingo::mat nz{{0.0, 5.0}, {7.0, 0.0}, {0.0, 8.0}};
  const auto idx = dingo::find(nz);
  CHECK(idx.n_elem() == 3);
  const auto vals = dingo::elem(nz, idx);
  CHECK(vals(0) == Catch::Approx(7.0));
  CHECK(vals(2) == Catch::Approx(8.0));

  dingo::cube z(2, 3, 1, dingo::fill::zeros);
  z(1, 2, 0) = 9.0;
  const auto sq = dingo::squeeze(z);
  CHECK(sq.n_rows() == 2);
  CHECK(sq.n_cols() == 3);
  CHECK(sq(1, 2) == Catch::Approx(9.0));

  dingo::cube pc(2, 3, 4, dingo::fill::zeros);
  const auto p = dingo::permute(pc, 2, 1, 3);
  CHECK(p.n_rows() == 3);
  CHECK(p.n_cols() == 2);
  CHECK(p.n_slices() == 4);

  const dingo::cx_mat cm{{{3.0, 4.0}, {1.0, -1.0}}};
  const auto cj = dingo::conj(cm);
  CHECK(cj(0, 0).imag() == Catch::Approx(-4.0));
  CHECK(dingo::real(cm)(0, 1) == Catch::Approx(1.0));
  CHECK(dingo::imag(cm)(0, 1) == Catch::Approx(-1.0));
  CHECK(dingo::abs2(cm)(0, 0) == Catch::Approx(25.0));
  CHECK(dingo::angle(dingo::cx_mat{{{0.0, 1.0}}})(0, 0) == Catch::Approx(1.57079632679).margin(1e-8));
}

TEST_CASE("Eigen interop adapters", "[core]") {
  Eigen::MatrixXd x(2, 2);
  x << 1.0, 2.0, 3.0, 4.0;
  auto m = dingo::from_eigen_mat(x);
  CHECK(m(1, 0) == Catch::Approx(3.0));

  auto& xe = dingo::as_eigen(m);
  xe(0, 1) = 9.0;
  CHECK(m(0, 1) == Catch::Approx(9.0));

  Eigen::VectorXd vc(3);
  vc << 5.0, 6.0, 7.0;
  const auto c = dingo::from_eigen_col(vc);
  CHECK(c.n_elem() == 3);
  CHECK(c(2) == Catch::Approx(7.0));

  Eigen::RowVectorXd vr(3);
  vr << 8.0, 9.0, 10.0;
  const auto r = dingo::from_eigen_row(vr);
  CHECK(r.n_elem() == 3);
  CHECK(r(1) == Catch::Approx(9.0));
}
