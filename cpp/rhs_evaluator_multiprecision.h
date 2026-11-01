// Sinc evaluation of f_2=(kappa^2-Delta)^s (1-r^2)_+^2 in d=2,3,
// using the direct approach in d=2,3 and the split approach in d=2,
// cf. Eq. (3.14) and Eq. (3.27). This implementation uses
// Boost multiprecision floating-point types.
//
// Wenyu Lei
// Oct. 26, 2026

#pragma once

#include <deal.II/base/exceptions.h>

#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/math/special_functions/gamma.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "radial_heat.h"

namespace RSOStudy::Multiprecision
{
  // Argument-dependent lookup supplies the Boost overloads for Number.
  using std::abs;
  using std::ceil;
  using std::cos;
  using std::exp;
  using std::expm1;
  using std::log;
  using std::pow;

  /** Convert decimal input without first rounding multiprecision to double. */
  template <typename Number>
  Number
  number_from_string(const std::string &text)
  {
    return Number(text);
  }

  // Decimal strings preserve the settings at every working precision.
  struct KernelParameters
  {
    // a in the strip |Im z|<a and the balanced truncation formulas.
    std::string strip_half_width = "1.4";
    // Section 4: t<cap and (1-r)^2>=separation_factor*t select the polynomial.
    std::string small_t_threshold = "0.001";
    std::string separation_factor = "480";
    // Additional numerical cutoffs: |v|<=32 and kappa^2*t<=70.
    std::string gaussian_cutoff    = "32";
    std::string direct_heat_cutoff = "70";
    // 0: balanced N^+/-; positive: exponential-tail target for the FEM test.
    // This target does not bound the complete pointwise error.
    std::string tail_tolerance = "0";
  };


  // Gauss-Legendre quadrature rule on [-1,1] in Number arithmetic.
  template <typename Number>
  class GaussLegendreRule
  {
  public:
    explicit GaussLegendreRule(const unsigned int order)
      : points(order)
      , weights(order)
    {
      AssertThrow(order >= 2,
                  dealii::ExcMessage("The Gauss order must be at least two."));

      const Number pi        = boost::math::constants::pi<Number>();
      const Number tolerance = 8 * std::numeric_limits<Number>::epsilon();
      for (unsigned int i = 0; i < (order + 1) / 2; ++i)
        {
          Number root      = cos(pi * (Number(i) + Number(0.75)) /
                            (Number(order) + Number(0.5)));
          bool   converged = false;
          for (unsigned int iteration = 0; iteration < 100; ++iteration)
            {
              const auto   value  = polynomial_and_derivative(order, root);
              const Number update = value.first / value.second;
              root -= update;
              if (abs(update) <= tolerance)
                {
                  converged = true;
                  break;
                }
            }
          AssertThrow(converged,
                      dealii::ExcMessage(
                        "A Gauss--Legendre root did not converge."));

          // xi_nu is a root of P_m; omega_nu=2/[(1-xi_nu^2)P_m'(xi_nu)^2].
          const Number derivative =
            polynomial_and_derivative(order, root).second;
          points[i]             = -root;
          points[order - 1 - i] = root;
          weights[i]            = weights[order - 1 - i] =
            2 / ((1 - root * root) * derivative * derivative);
        }
    }

    std::vector<Number> points;
    std::vector<Number> weights;

  private:
    static std::pair<Number, Number>
    polynomial_and_derivative(const unsigned int order, const Number &point)
    {
      Number previous = 1;
      Number current  = point;
      // Three-term Legendre recurrence, starting from P_0=1 and P_1=x.
      for (unsigned int k = 2; k <= order; ++k)
        {
          const Number next =
            ((2 * k - 1) * point * current - (k - 1) * previous) / k;
          previous = current;
          current  = next;
        }
      const Number derivative =
        order * (point * current - previous) / (point * point - 1);
      return {current, derivative};
    }
  };


  /**
   * The two hypergeometric series needed for the massless n=d=2 formula.
   * Here c-a-b=2-2s>0. Gauss's formula evaluates the endpoint z=1.
   * After the first terms, successive absolute terms have ratio at most z;
   * the factor 1-z therefore controls the omitted geometric tail.
   */
  template <typename Number>
  Number
  hypergeometric_2f1(const Number &a,
                     const Number &b,
                     const Number &c,
                     const Number &z)
  {
    using boost::math::tgamma;
    if (z == 1)
      return tgamma(c) * tgamma(c - a - b) / (tgamma(c - a) * tgamma(c - b));
    if (z == 0)
      return Number(1);

    const Number tolerance = std::numeric_limits<Number>::epsilon() / 8;
    Number       term      = 1;
    Number       sum       = 1;
    for (unsigned int k = 0; k < 1000000; ++k)
      {
        term *= (a + k) * (b + k) * z / ((c + k) * (k + 1));
        sum += term;
        if (k >= 2 && abs(term) <= tolerance * (1 - z) *
                                     std::max(Number(1), Number(abs(sum))))
          return sum;
      }
    AssertThrow(
      false,
      dealii::ExcMessage(
        "The hypergeometric series did not converge; r is too close to one."));
  }


  // Equation (3.24), specialized to n=d=2: b_2=(-Delta)^s u_2.
  template <typename Number>
  Number
  massless_rhs(const Number &r, const Number &s)
  {
    using boost::math::tgamma;
    if (r <= 1)
      return pow(Number(4), s) * 2 * tgamma(1 + s) / tgamma(3 - s) *
             hypergeometric_2f1<Number>(1 + s, s - 2, Number(1), r * r);

    const Number c_s = s / tgamma(1 - s);
    return -c_s * pow(Number(4), s) * tgamma(1 + s) / 3 * pow(r, -2 - 2 * s) *
           hypergeometric_2f1<Number>(1 + s, 1 + s, Number(4), 1 / (r * r));
  }


  template <typename Number>
  struct Evaluation
  {
    // f_2; in the split scheme rhs=massless+correction=b_2+R_2.
    // The direct scheme leaves massless and correction at zero.
    Number rhs        = 0;
    Number massless   = 0;
    Number correction = 0;
    // N^- and N^+ in the sum over -N^- <= j <= N^+.
    int n_minus = 0;
    int n_plus  = 0;
  };


  /**
   * Direct and massless-plus-correction sinc schemes from Section 3.
   * Number is a Boost multiprecision floating-point type.
   * The direct scheme supports d=2,3; the massless formula supplied here is
   * for d=2. Both accept kappa>0. Multiprecision nodes and intermediate heat
   * values are computed entirely in the selected precision.
   */
  template <typename Number>
  class Evaluator
  {
  public:
    using number_type = Number;

    explicit Evaluator(const unsigned int      order,
                       const KernelParameters &parameters = {},
                       const unsigned int      dimension  = 2,
                       const Number           &kappa      = Number(1))
      : radial_rule(order)
      , dimension(dimension)
      , kappa_squared(kappa * kappa)
      , strip_half_width(
          number_from_string<Number>(parameters.strip_half_width))
      , small_t_threshold(
          number_from_string<Number>(parameters.small_t_threshold))
      , separation_factor(
          number_from_string<Number>(parameters.separation_factor))
      , gaussian_cutoff(number_from_string<Number>(parameters.gaussian_cutoff))
      , direct_heat_cutoff(
          number_from_string<Number>(parameters.direct_heat_cutoff))
      , tail_tolerance(number_from_string<Number>(parameters.tail_tolerance))
    {
      AssertThrow(dimension == 2 || dimension == 3,
                  dealii::ExcMessage("Require dimension 2 or 3."));
      AssertThrow(boost::math::isfinite(kappa) && kappa > 0 &&
                    boost::math::isfinite(kappa_squared) && kappa_squared > 0,
                  dealii::ExcMessage(
                    "Kappa and its square must be finite and positive."));
      AssertThrow(boost::math::isfinite(tail_tolerance) &&
                    tail_tolerance >= 0 && tail_tolerance < 1,
                  dealii::ExcMessage("Tail tolerance must lie in [0,1)."));
      const Number pi = boost::math::constants::pi<Number>();
      for (const Number *value : {&strip_half_width,
                                  &small_t_threshold,
                                  &separation_factor,
                                  &gaussian_cutoff,
                                  &direct_heat_cutoff})
        AssertThrow(
          boost::math::isfinite(*value) && *value > 0,
          dealii::ExcMessage(
            "Kernel parameters must be finite and strictly positive."));
      AssertThrow(strip_half_width < pi / 2,
                  dealii::ExcMessage(
                    "The strip half-width must be below pi/2."));
      // The common kernel already discards |v|>32. A smaller optional cutoff
      // is supported here; increasing it needs a corresponding kernel change.
      AssertThrow(gaussian_cutoff <= 32,
                  dealii::ExcMessage(
                    "The Gaussian cutoff must not exceed 32."));
    }


    // Approximate P_t u_2. At t=t_j, radial_gauss_heat gives mathcal
    // H_{j,m}^{(d)}.
    Number
    heat(const Number &t, const Number &r) const
    {
      AssertThrow(boost::math::isfinite(t) && t > 0 &&
                    boost::math::isfinite(r) && r >= 0,
                  dealii::ExcMessage("Require finite t>0 and r>=0."));
      const Number delta = abs(1 - r);
      if (r < 1 && t < small_t_threshold &&
          delta * delta >= separation_factor * t)
        {
          // Section 4 uses the full identity P_t p=p+t Delta p+t^2 Delta^2 p/2
          // for p=(1-r^2)^2 on R^d. Only the zero-extension tail is omitted;
          // this is more than the first-order approximation in Remark 3.1.
          const Number u = pow((1 - r) * (1 + r), 2);
          return u + t * polynomial_laplacian(r) +
                 half_polynomial_bilaplacian() * t * t;
        }
      // Exterior point: the same separation bound makes the heat tail
      // negligible.
      if (r > 1 && delta * delta >= separation_factor * t)
        return Number(0);

      return radial_gauss_heat(t, r);
    }


    // The bracket w(t,r) in the direct scheme (3.14).
    Number
    direct_bracket(const Number &t, const Number &r) const
    {
      const Number u = r < 1 ? Number(pow((1 - r) * (1 + r), 2)) : Number(0);
      // Preserve the established evaluator's negligible damped-heat cutoff.
      const Number scaled_time = kappa_squared * t;
      if (scaled_time > direct_heat_cutoff)
        return u;

      const Number delta       = abs(1 - r);
      const Number attenuation = exp(-scaled_time);
      if (r < 1 && t < small_t_threshold &&
          delta * delta >= separation_factor * t)
        // Section 4's w_c: form 1-exp(-kappa^2*t) with -expm1(...).
        return -expm1(-scaled_time) * u -
               attenuation * t *
                 (polynomial_laplacian(r) + half_polynomial_bilaplacian() * t);
      return u - attenuation * heat(t, r);
    }


    // Choose N^-/N^+ using (3.21) or (3.30), then evaluate the sinc sum.
    Evaluation<Number>
    evaluate(const Number &r,
             const Number &s,
             const Number &sinc_step,
             const bool    split) const
    {
      validate_arguments(r, s, sinc_step);
      const Number two_pi_a =
        2 * boost::math::constants::pi<Number>() * strip_half_width;
      // The supplied split formula has d=2, hence heat decay contributes 1.
      const Number positive_decay = split ? Number(s + 1) : s;
      Number       n_minus_bound, n_plus_bound;
      if (tail_tolerance == 0)
        {
          // Balance both tails against exp(-2*pi*a/l).
          n_minus_bound = ceil(two_pi_a / ((1 - s) * sinc_step * sinc_step));
          n_plus_bound =
            ceil(two_pi_a / (positive_decay * sinc_step * sinc_step));
        }
      else
        {
          // Instead prescribe exp(-decay*N*l)<=tail_tolerance.
          n_minus_bound = ceil(-log(tail_tolerance) / ((1 - s) * sinc_step));
          n_plus_bound =
            ceil(-log(tail_tolerance) / (positive_decay * sinc_step));
        }
      AssertThrow(boost::math::isfinite(n_minus_bound) &&
                    boost::math::isfinite(n_plus_bound) &&
                    n_minus_bound <= std::numeric_limits<int>::max() &&
                    n_plus_bound < std::numeric_limits<int>::max(),
                  dealii::ExcMessage(
                    "The requested sinc index exceeds int range."));
      return evaluate(r,
                      s,
                      sinc_step,
                      static_cast<int>(n_minus_bound),
                      static_cast<int>(n_plus_bound),
                      split);
    }


    // The paper's outer rule, with l=sinc_step and explicit N^-,N^+.
    // split=false gives (3.14); split=true gives (3.27).
    Evaluation<Number>
    evaluate(const Number &r,
             const Number &s,
             const Number &sinc_step,
             const int     n_minus,
             const int     n_plus,
             const bool    split) const
    {
      validate_arguments(r, s, sinc_step);
      AssertThrow(!split || dimension == 2,
                  dealii::ExcMessage(
                    "The supplied split formula supports dimension 2 only."));
      AssertThrow(n_minus >= 0 && n_plus >= 0 &&
                    n_plus < std::numeric_limits<int>::max(),
                  dealii::ExcMessage("Invalid sinc truncation indices."));

      Evaluation<Number> result;
      result.n_minus = n_minus;
      result.n_plus  = n_plus;
      if (split)
        result.massless = massless_rhs(r, s);

      Number sinc_sum = 0;
      for (int j = -n_minus; j <= n_plus; ++j)
        {
          const Number y_j = j * sinc_step;
          const Number t_j = exp(y_j);
          AssertThrow(boost::math::isfinite(t_j) && t_j > 0,
                      dealii::ExcMessage(
                        "A quadrature parameter t_j is outside the working "
                        "floating-point range."));
          Number increment;
          if (split)
            // R_2 increment: (1-exp(-kappa^2*t_j)) times the computed heat.
            increment = shift_bracket(t_j, r);
          else
            // Direct increment: w(t_j,r), with the stable branch for small t.
            increment = direct_bracket(t_j, r);

          const Number weighted_increment = exp(-s * y_j) * increment;
          sinc_sum += weighted_increment;
        }
      const Number c_s           = s / boost::math::tgamma(1 - s);
      const Number sinc_integral = c_s * sinc_step * sinc_sum;
      if (split)
        result.correction = sinc_integral;
      result.rhs = result.massless + sinc_integral;
      return result;
    }

  private:
    // Equation (3.13) on the Remark 3.2 partition, represented in
    // v=(q-r)/(2*sqrt(t)). The rule stores xi_nu,omega_nu on [-1,1], hence
    // the half-width in its mapped weight. The common heat_integrand
    // already includes dq/dv=2*sqrt(t); do not multiply by it again.
    Number
    radial_gauss_heat(const Number &t, const Number &r) const
    {
      const auto panels   = manuscript_numerics::radial_partition<Number>(r, t);
      Number     heat_sum = 0;
      for (std::size_t panel = 1; panel < panels.size(); ++panel)
        {
          if (panels[panel - 1] >= gaussian_cutoff ||
              panels[panel] <= -gaussian_cutoff)
            continue;
          const Number lower =
            std::max(panels[panel - 1], Number(-gaussian_cutoff));
          const Number upper      = std::min(panels[panel], gaussian_cutoff);
          const Number half_width = (upper - lower) / 2;
          const Number midpoint   = (upper + lower) / 2;
          for (std::size_t i = 0; i < radial_rule.points.size(); ++i)
            {
              const Number v = midpoint + half_width * radial_rule.points[i];
              const Number gauss_weight = half_width * radial_rule.weights[i];
              heat_sum +=
                gauss_weight *
                manuscript_numerics::heat_integrand<Number>(dimension, r, t, v);
            }
        }
      return heat_sum;
    }

    // The heat factor in R_2, (3.25); expm1 preserves its small-t value.
    Number
    shift_bracket(const Number &t, const Number &r) const
    {
      return Number(-expm1(-kappa_squared * t) * heat(t, r));
    }

    // P_t p = p + t Delta p + (t^2/2) Delta^2 p for p=(1-r^2)^2.
    Number
    polynomial_laplacian(const Number &r) const
    {
      return -Number(4 * dimension) + 4 * (dimension + 2) * r * r;
    }

    // One half of Delta^2 p: the coefficient of t^2 in P_t p.
    Number
    half_polynomial_bilaplacian() const
    {
      return Number(4 * dimension * (dimension + 2));
    }

    static void
    validate_arguments(const Number &r, const Number &s, const Number &step)
    {
      AssertThrow(boost::math::isfinite(r) && boost::math::isfinite(s) &&
                    boost::math::isfinite(step) && r >= 0 && s > 0 && s < 1 &&
                    step > 0,
                  dealii::ExcMessage(
                    "Require finite r>=0, 0<s<1, and step>0."));
    }

    GaussLegendreRule<Number> radial_rule;
    const unsigned int        dimension;
    const Number              kappa_squared;
    const Number              strip_half_width;
    const Number              small_t_threshold;
    const Number              separation_factor;
    const Number              gaussian_cutoff;
    const Number              direct_heat_cutoff;
    const Number              tail_tolerance;
  };
} // namespace RSOStudy::Multiprecision
