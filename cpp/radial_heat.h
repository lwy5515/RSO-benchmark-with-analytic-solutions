/**
 * Radial heat evaluation
 * (P_t u_2)(r) = integral_0^1 (1-q^2)^2 H_d(t;r,q) dq
 * for the case n = 2.
 *
 * Subroutines :
 * radial_partition(): Remark 3.2 and the implementation in Section 4.
 * scaled_i0(): Remark 3.3, tilde I_0(eta) = exp(-eta) I_0(eta).
 * heat_integrand(): H_2 in (3.10), H_3 in (3.12), and the sum in (3.13).
 *
 * Wenyu Lei
 * Oct. 26, 2026
 */

#pragma once

#include <boost/math/constants/constants.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>


namespace manuscript_numerics
{
  /**
   * Construct the partition of [0,1] in Remark 3.2, expressed in v.
   * Physical endpoints q=0,1 become v=-r/(2 sqrt(t)), (1-r)/(2 sqrt(t)).
   */
  template <class Real>
  std::vector<Real>
  radial_partition(const Real &r, const Real &t)
  {
    using std::sqrt;
    const Real sqrt_t  = sqrt(t);
    const Real v_begin = -r / (2 * sqrt_t);
    const Real v_end   = (1 - r) / (2 * sqrt_t);

    std::vector<Real> v_points{v_begin, v_end};
    const auto        add_interior_point = [&](const Real &v) {
      if (v > v_begin && v < v_end)
        v_points.push_back(v);
    };

    // Inserts the peak q=r when it lies in (0,1).
    add_interior_point(Real(0));

    // q=r +/- 2^k sqrt(t), k=1,2,... corresponds to v=+/-1,+/-2,+/-4,... .
    // Stop once the two candidate q values enclose the physical interval.
    for (Real half_width_v = 1;; half_width_v *= 2)
      {
        add_interior_point(-half_width_v);
        add_interior_point(half_width_v);
        if (-half_width_v <= v_begin && half_width_v >= v_end)
          break;
      }

    // Exterior point: the closest point in the support is q=1. Resolve its
    // physical scale min{sqrt(t), t/(r-1)} with q=1-2^k*endpoint_width_q.
    if (r > 1)
      {
        const Real endpoint_width_q = std::min(Real(sqrt_t), Real(t / (r - 1)));
        const Real endpoint_width_v = endpoint_width_q / (2 * sqrt_t);
        for (Real offset_v = endpoint_width_v; offset_v < v_end - v_begin;
             offset_v *= 2)
          add_interior_point(Real(v_end - offset_v));
      }

    std::sort(v_points.begin(), v_points.end());
    v_points.erase(std::unique(v_points.begin(), v_points.end()),
                   v_points.end());
    return v_points;
  }

  /**
   * Remark 3.3: evaluate tilde I_0(eta) = exp(-eta) I_0(eta), eta>=0.
   * The switch eta=60 follows the implementation described in Section 4.
   * The stopping tolerance follows the working precision. The asymptotic
   * recurrence continues beyond the six displayed terms when needed.
   * Here we note that for double precision, the GSL function
   * gsl_sf_bessel_I0_scaled(eta) is equivalent to tilde I_0(eta).
   */
  template <class Real>
  Real
  scaled_i0(const Real &eta)
  {
    using std::exp;
    using std::sqrt;
    const Real relative_tolerance = std::numeric_limits<Real>::epsilon() / 4;

    constexpr int maximum_series_terms = 10000;
    constexpr int asymptotic_switch    = 60;

    Real term = 1;
    Real sum  = 1;
    if (eta < asymptotic_switch)
      {
        // I_0(eta) = sum_{k=0}^infinity (eta^2/4)^k/(k!)^2.
        // term_k = term_{k-1} * (eta^2/4)/k^2, with term_0=1.
        const Real eta_squared_over_four = eta * eta / 4;
        for (int k = 1; k < maximum_series_terms; ++k)
          {
            term *= eta_squared_over_four / Real(k * k);
            sum += term;
            if (term <= relative_tolerance * sum)
              break;
          }
        return exp(-eta) * sum;
      }

    // tilde I_0(eta) ~ (1 + 1/(8 eta) + 9/(128 eta^2) + ...)/sqrt(2 pi eta).
    // term_k = term_{k-1} * (2k-1)^2/(8 k eta), with term_0=1.
    // This is an asymptotic series: stop before the terms begin to grow.
    for (int k = 1; k < maximum_series_terms; ++k)
      {
        const Real odd_index = 2 * k - 1;
        const Real next_term = term * odd_index * odd_index / (8 * k * eta);
        if (next_term > term)
          break;
        sum += next_term;
        term = next_term;
        if (term <= relative_tolerance * sum)
          break;
      }
    return sum / sqrt(2 * boost::math::constants::pi<Real>() * eta);
  }

  /**
   * Return (1-q^2)^2 H_d(t;r,q) dq/dv for d=2 or d=3.
   *
   * The scaled kernels in (3.10) and (3.12) are
   *   H_2 = q/(2t) exp(-v^2) tilde I_0(eta),  eta = rq/(2t),
   *   H_3 = q^2/(2 sqrt(pi) t^(3/2)) exp(-v^2) Phi(zeta),  zeta = rq/t,
   * where Phi(zeta) = -expm1(-zeta)/zeta and Phi(0)=1.
   *
   * Multiplying by dq/dv=2 sqrt(t) gives the prefactors used below.
   * Production quadrature uses q in [0,1]. The polynomial is left unclipped
   * so the independent check can also test whole-space polynomial moments.
   */
  template <class Real>
  Real
  heat_integrand(const int   dimension,
                 const Real &r,
                 const Real &t,
                 const Real &v)
  {
    using std::exp;
    using std::expm1;
    using std::sqrt;

    // Additional numerical tail truncation: omit |v|>32, where the Gaussian
    // factor is below exp(-1024). Callers also skip panels beyond this cutoff.
    constexpr int gaussian_cutoff_squared = 32 * 32;
    if (v * v > gaussian_cutoff_squared)
      return Real(0);

    // Change of variables from the physical radius q to the coordinate v.
    const Real sqrt_t    = sqrt(t);
    const Real q_minus_r = 2 * sqrt_t * v;
    const Real q         = r + q_minus_r;
    const Real gaussian  = exp(-v * v);

    // For n=2, (1-q^2)^2 = (1-q)^2 (1+q)^2. Near q=1, q itself can round
    // to 1; evaluating 1-q=(1-r)-(q-r) retains the small boundary offset.
    const Real one_minus_q = (1 - r) - q_minus_r;
    const Real bump        = one_minus_q * one_minus_q * (1 + q) * (1 + q);
    const Real eta         = r * q / (2 * t);

    if (dimension == 2)
      {
        // (1-q^2)^2 H_2 dq/dv:
        // q/(2t) * 2 sqrt(t) = q/sqrt(t).
        const Real scaled_bessel = scaled_i0<Real>(eta);
        return bump * q / sqrt_t * gaussian * scaled_bessel;
      }
    else if (dimension == 3)
      {
        // (1-q^2)^2 H_3 dq/dv:
        // q^2/(2 sqrt(pi) t^(3/2)) * 2 sqrt(t) = q^2/(sqrt(pi) t).
        // zeta=2*eta=rq/t; expm1 gives an accurate Phi near zeta=0.
        const Real zeta = 2 * eta;
        const Real phi  = zeta == 0 ? Real(1) : Real(-expm1(-zeta) / zeta);
        return bump * q * q / (sqrt(boost::math::constants::pi<Real>()) * t) *
               gaussian * phi;
      }
    else
      {
        AssertThrow(false, dealii::ExcMessage("Require dimension 2 or 3."));
        return Real(0); // Unreachable, but avoids compiler warnings.
      }
  }
} // namespace manuscript_numerics
