// Sinc evaluation of f_2=(kappa^2-Delta)^s (1-r^2)_+^2 in d=2,3,
// using the direct approach in d=2,3 and the split approach in d=2,
// cf. Eq. (3.14) and Eq. (3.27). This implementation uses
// double precision.
//
// Wenyu Lei
// Oct. 26, 2026

#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/quadrature_lib.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_bessel.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include "radial_heat.h"

namespace RSOStudy
{
  // A zero tail_tolerance selects the balanced sinc indices from Section 3.
  // A positive value selects an exponential-tail target, not a total error
  // bound.
  struct KernelParameters
  {
    double strip_half_width   = 1.4;   // a in the balanced sinc indices
    double small_t_threshold  = 0.001; // cap in t_0(r)
    double separation_factor  = 480;   // t_0(r) also uses (1-r)^2/480
    double gaussian_cutoff    = 32;    // retain |v| <= 32
    double direct_heat_cutoff = 70;    // omit damped heat for kappa^2*t > 70
    double tail_tolerance     = 0;
  };

  struct Evaluation
  {
    // rhs approximates f_2. Only the split scheme fills b_2 and R_2 below.
    double rhs = 0, massless = 0, correction = 0;
    // N^- and N^+ are the retained negative and positive sinc indices.
    int n_minus = 0, n_plus = 0;
  };

  // We directly implement the hypergeometric series 2F1(a,b;c;z) for |z|<1.
  // Note that GSL functions may not be available in all environments, so we
  // avoid them here.
  inline double
  hypergeometric_2f1(const double a,
                     const double b,
                     const double c,
                     const double z)
  {
    if (z == 1)
      return std::tgamma(c) * std::tgamma(c - a - b) /
             (std::tgamma(c - a) * std::tgamma(c - b));
    if (z == 0)
      return 1.;

    const double tolerance = std::numeric_limits<double>::epsilon() / 8;
    double       term = 1, sum = 1;
    for (unsigned int k = 0; k < 1000000; ++k)
      {
        term *= (a + k) * (b + k) * z / ((c + k) * (k + 1));
        sum += term;
        if (k >= 2 &&
            std::abs(term) <= tolerance * (1 - z) * std::max(1., std::abs(sum)))
          return sum;
      }
    AssertThrow(
      false,
      dealii::ExcMessage(
        "The hypergeometric series did not converge; r is too close to one."));
  }

  // Equation (3.24), specialized to n=d=2; r=1 uses the continuous limit.
  inline double
  massless_rhs(const double r, const double s)
  {
    if (r <= 1)
      return std::pow(4., s) * 2 * std::tgamma(1 + s) / std::tgamma(3 - s) *
             hypergeometric_2f1(1 + s, s - 2, 1., r * r);
    return -s / std::tgamma(1 - s) * std::pow(4., s) * std::tgamma(1 + s) / 3 *
           std::pow(r, -2 - 2 * s) *
           hypergeometric_2f1(1 + s, 1 + s, 4., 1 / (r * r));
  }

  // Implementation of sinc formulas for n=2; constructor order=m, dimension=d.
  class Evaluator
  {
  public:
    using number_type = double;

    explicit Evaluator(const unsigned int      order,
                       const KernelParameters &parameters = {},
                       const unsigned int      dimension  = 2,
                       const double            kappa      = 1.)
      : radial_rule(checked_order(order))
      , parameters(parameters)
      , dimension(dimension)
      , kappa_squared(kappa * kappa)
    {
      AssertThrow(dimension == 2 || dimension == 3,
                  dealii::ExcMessage("Require dimension 2 or 3."));
      AssertThrow(std::isfinite(kappa) && kappa > 0 &&
                    std::isfinite(kappa_squared) && kappa_squared > 0,
                  dealii::ExcMessage(
                    "Kappa and its square must be finite and positive."));
      for (const double value : {parameters.strip_half_width,
                                 parameters.small_t_threshold,
                                 parameters.separation_factor,
                                 parameters.gaussian_cutoff,
                                 parameters.direct_heat_cutoff})
        AssertThrow(std::isfinite(value) && value > 0,
                    dealii::ExcMessage(
                      "Kernel parameters must be finite and positive."));
      AssertThrow(parameters.strip_half_width < dealii::numbers::PI / 2,
                  dealii::ExcMessage(
                    "The strip half-width must be below pi/2."));
      AssertThrow(parameters.gaussian_cutoff <= 32,
                  dealii::ExcMessage(
                    "The Gaussian cutoff must not exceed 32."));
      AssertThrow(std::isfinite(parameters.tail_tolerance) &&
                    parameters.tail_tolerance >= 0 &&
                    parameters.tail_tolerance < 1,
                  dealii::ExcMessage("Tail tolerance must lie in [0,1)."));
    }

    // Heat action used in both sinc schemes. For small t at interior points,
    // Section 4 uses P_t p=p+t*Delta p+(t^2/2)*Delta^2 p instead of
    // quadrature, with p(r)=(1-r^2)^2 on the whole space. The missing
    // exterior contribution is exponentially small in delta^2/t. The test
    // below keeps t<cap and separation_factor*t<=delta^2, defining t_0(r).
    double
    heat(const double t, const double r) const
    {
      AssertThrow(std::isfinite(t) && t > 0 && std::isfinite(r) && r >= 0,
                  dealii::ExcMessage("Require finite t>0 and r>=0."));
      const double delta = std::abs(1 - r);
      if (r < 1 && t < parameters.small_t_threshold &&
          delta * delta >= parameters.separation_factor * t)
        return polynomial_value(r) + t * polynomial_laplacian(r) +
               half_polynomial_bilaplacian() * t * t;
      // Exterior point: the same separation bound makes the heat tail
      // negligible.
      if (r > 1 && delta * delta >= parameters.separation_factor * t)
        return 0.;

      return radial_gauss_heat(t, r);
    }

    // Direct-scheme increment w(t,r). The branch for small t uses Section 4's
    // full polynomial expression w_c, motivated by Remark 3.1; it is not
    // merely the leading term t*(kappa^2*u_2-Delta u_2).
    double
    direct_bracket(const double t, const double r) const
    {
      const double u_2         = r < 1 ? polynomial_value(r) : 0.;
      const double scaled_time = kappa_squared * t;
      if (scaled_time > parameters.direct_heat_cutoff)
        return u_2;
      const double delta       = std::abs(1 - r);
      const double attenuation = std::exp(-scaled_time);
      if (r < 1 && t < parameters.small_t_threshold &&
          delta * delta >= parameters.separation_factor * t)
        return -std::expm1(-scaled_time) * u_2 -
               attenuation * t *
                 (polynomial_laplacian(r) + half_polynomial_bilaplacian() * t);
      return u_2 - attenuation * heat(t, r);
    }

    // Select N^- and N^+. Balanced indices follow (3.21) for the direct
    // scheme and (3.30) for the split scheme. Its positive-tail exponent
    // is s+d/2=s+1 because the supplied massless formula has d=2.
    Evaluation
    evaluate(const double r,
             const double s,
             const double sinc_step,
             const bool   split) const
    {
      validate_arguments(r, s, sinc_step);
      const double two_pi_a =
        2 * dealii::numbers::PI * parameters.strip_half_width;
      const double positive_decay = split ? s + 1 : s;
      double       n_minus_bound, n_plus_bound;
      if (parameters.tail_tolerance == 0)
        {
          // Balance the discretization scale exp(-2*pi*a/l) with both tails.
          n_minus_bound =
            std::ceil(two_pi_a / ((1 - s) * sinc_step * sinc_step));
          n_plus_bound =
            std::ceil(two_pi_a / (positive_decay * sinc_step * sinc_step));
        }
      else
        {
          // Alternative used for the FEM loads: target each exponential tail.
          n_minus_bound = std::ceil(-std::log(parameters.tail_tolerance) /
                                    ((1 - s) * sinc_step));
          n_plus_bound  = std::ceil(-std::log(parameters.tail_tolerance) /
                                   (positive_decay * sinc_step));
        }
      AssertThrow(std::isfinite(n_minus_bound) && std::isfinite(n_plus_bound) &&
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

    // Equations (3.14) and (3.27), with explicit truncation indices.
    // The direct scheme integrates w; the split scheme integrates the
    // shift correction and adds the explicit massless term b_2.
    Evaluation
    evaluate(const double r,
             const double s,
             const double sinc_step,
             const int    n_minus,
             const int    n_plus,
             const bool   split) const
    {
      validate_arguments(r, s, sinc_step);
      AssertThrow(!split || dimension == 2,
                  dealii::ExcMessage(
                    "The supplied split formula supports dimension 2 only."));
      AssertThrow(n_minus >= 0 && n_plus >= 0 &&
                    n_plus < std::numeric_limits<int>::max(),
                  dealii::ExcMessage("Invalid sinc truncation indices."));
      Evaluation result;
      result.n_minus = n_minus;
      result.n_plus  = n_plus;
      if (split)
        result.massless = massless_rhs(r, s);

      double sinc_sum = 0, rounding_correction = 0;
      for (int j = -n_minus; j <= n_plus; ++j)
        {
          const double y_j = j * sinc_step;
          const double t_j = std::exp(y_j);
          AssertThrow(std::isfinite(t_j) && t_j > 0,
                      dealii::ExcMessage(
                        "A quadrature parameter t_j is outside the working "
                        "floating-point range."));
          double increment;
          if (split)
            increment = shift_bracket(t_j, r); // (1-exp(-kappa^2*t_j))*P_tj u_2
          else
            increment = direct_bracket(t_j, r); // w(t_j,r)

          // t_j^(-s)=exp(-s*y_j); compensate rounding in the outer sum.
          const double adjusted =
            std::exp(-s * y_j) * increment - rounding_correction;
          const double next   = sinc_sum + adjusted;
          rounding_correction = (next - sinc_sum) - adjusted;
          sinc_sum            = next;
        }
      const double c_s           = s / std::tgamma(1 - s);
      const double sinc_integral = c_s * sinc_step * sinc_sum;
      if (split)
        result.correction = sinc_integral; // approximation of R_2
      result.rhs = result.massless + sinc_integral;
      AssertThrow(std::isfinite(result.rhs),
                  dealii::ExcMessage("Nonfinite f_2 value."));
      return result;
    }

  private:
    // The approximation mathcal H_{j,m}^{(d)} in (3.13), written in
    // v=(q-r)/(2*sqrt(t)) using Remark 3.2's partition.
    // QGauss stores nodes on [0,1]. Multiplying by the v-panel width gives
    // its Gauss weight; radial_heat_integrand already contains dq/dv.
    double
    radial_gauss_heat(const double t, const double r) const
    {
      const auto v_panels = manuscript_numerics::radial_partition(r, t);
      double     heat_sum = 0;
      for (std::size_t panel = 1; panel < v_panels.size(); ++panel)
        {
          const double lower =
            std::max(v_panels[panel - 1], -parameters.gaussian_cutoff);
          const double upper =
            std::min(v_panels[panel], parameters.gaussian_cutoff);
          if (lower >= upper)
            continue;
          const double width = upper - lower;
          for (unsigned int i = 0; i < radial_rule.size(); ++i)
            {
              const double v = lower + width * radial_rule.point(i)[0];
              const double gauss_weight = width * radial_rule.weight(i);
              heat_sum += gauss_weight * radial_heat_integrand(r, t, v);
            }
        }
      return heat_sum;
    }

    // Shift-correction increment in (3.25)/(3.27). expm1 accurately forms
    // 1-exp(-kappa^2*t) without subtracting two nearly equal numbers.
    double
    shift_bracket(const double t, const double r) const
    {
      return -std::expm1(-kappa_squared * t) * heat(t, r);
    }

    // (1-q^2)^2 H_d(t;r,q) dq/dv, with v=(q-r)/(2 sqrt(t)).
    // H_2 is (3.10), H_3 is (3.12), and dq/dv=2*sqrt(t).
    double
    radial_heat_integrand(const double r, const double t, const double v) const
    {
      if (v * v > 32 * 32)
        return 0.;
      const double sqrt_t    = std::sqrt(t);
      const double q_minus_r = 2 * sqrt_t * v;
      const double q         = r + q_minus_r;
      // Preserve the distance to q=1 when q itself would round to one.
      const double one_minus_q = (1 - r) - q_minus_r;
      const double polynomial  = one_minus_q * one_minus_q * (1 + q) * (1 + q);
      const double gaussian    = std::exp(-v * v);
      const double eta         = r * q / (2 * t);
      if (dimension == 2)
        {
          // H_2*dq/dv = q/sqrt(t) * exp(-v^2) * tilde I_0(eta).
          // Remark 3.3: GSL returns exp(-eta) I_0(eta) directly.
          gsl_sf_result bessel;
          const int     status = gsl_sf_bessel_I0_scaled_e(eta, &bessel);
          AssertThrow(status == GSL_SUCCESS && std::isfinite(bessel.val),
                      dealii::ExcMessage(std::string("GSL scaled I_0: ") +
                                         gsl_strerror(status)));
          return polynomial * q / sqrt_t * gaussian * bessel.val;
        }
      else if (dimension == 3)
        {
          // H_3*dq/dv = q^2/(sqrt(pi)*t) * exp(-v^2) * Phi(r*q/t).
          const double zeta = 2 * eta;
          const double phi  = zeta == 0 ? 1. : -std::expm1(-zeta) / zeta;
          return polynomial * q * q / (std::sqrt(dealii::numbers::PI) * t) *
                 gaussian * phi;
        }
      else
        {
          AssertThrow(false,
                      dealii::ExcMessage(
                        "Unsupported dimension for radial heat integrand."));
          return 0;
        }
    }

    // Unrestricted p(r), not the zero-extended u_2 outside the unit ball.
    static double
    polynomial_value(const double r)
    {
      return std::pow((1 - r) * (1 + r), 2);
    }

    // P_t p = p + t Delta p + (t^2/2) Delta^2 p for p=(1-r^2)^2.
    double
    polynomial_laplacian(const double r) const
    {
      return -4. * dimension + 4 * (dimension + 2) * r * r;
    }

    // One half of Delta^2 p: the coefficient of t^2 in P_t p.
    double
    half_polynomial_bilaplacian() const
    {
      return 4. * dimension * (dimension + 2);
    }

    static unsigned int
    checked_order(const unsigned int order)
    {
      AssertThrow(order >= 2,
                  dealii::ExcMessage("The Gauss order must be at least two."));
      return order;
    }

    static void
    validate_arguments(const double r, const double s, const double step)
    {
      AssertThrow(std::isfinite(r) && std::isfinite(s) && std::isfinite(step) &&
                    r >= 0 && s > 0 && s < 1 && step > 0,
                  dealii::ExcMessage(
                    "Require finite r>=0, 0<s<1, and step>0."));
    }

    dealii::QGauss<1>      radial_rule;
    const KernelParameters parameters;
    const unsigned int     dimension;
    const double           kappa_squared;
  };
} // namespace RSOStudy
