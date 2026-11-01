#include <deal.II/base/exceptions.h>
#include <deal.II/base/timer.h>

#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>

#include <gsl/gsl_errno.h>

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#include "parameters.h"
#include "rhs_evaluator.h"
#include "rhs_evaluator_multiprecision.h"

namespace RSOF2
{
  template <unsigned int bits>
  using Number = boost::multiprecision::number<
    boost::multiprecision::cpp_bin_float<bits,
                                         boost::multiprecision::digit_base_2>>;

  // Double-precision example: construct an evaluator, then call it at each r.
  void
  evaluate_double(const Parameters &parameters)
  {
    RSOStudy::KernelParameters kernel;
    if (parameters.truncation == "tail tolerance")
      kernel.tail_tolerance = std::stod(parameters.tail_tolerance);

    const RSOStudy::Evaluator evaluator(parameters.order,
                                        kernel,
                                        parameters.dimension,
                                        std::stod(parameters.kappa));
    const double              s = std::stod(parameters.exponent);
    const double sinc_step      = std::stod(parameters.step); // l in the paper
    const bool   use_split      = parameters.method == "split";

    std::cout << std::scientific
              << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (const auto &radius : parameters.radii)
      {
        const double r  = std::stod(radius);
        const double f2 = evaluator.evaluate(r, s, sinc_step, use_split).rhs;
        std::cout << "r = " << radius << ", f_2(r) = " << f2 << '\n';
      }
  }

  // Same example in higher precision; convert decimal strings directly to Real.
  template <typename Real>
  void
  evaluate_multiprecision(const Parameters &parameters)
  {
    namespace MP = RSOStudy::Multiprecision;
    MP::KernelParameters kernel;
    if (parameters.truncation == "tail tolerance")
      kernel.tail_tolerance = parameters.tail_tolerance;

    const MP::Evaluator<Real> evaluator(parameters.order,
                                        kernel,
                                        parameters.dimension,
                                        Real(parameters.kappa));
    const Real                s(parameters.exponent);
    const Real                sinc_step(parameters.step); // l in the paper
    const bool                use_split = parameters.method == "split";

    std::cout << std::scientific
              << std::setprecision(std::numeric_limits<Real>::max_digits10);
    for (const auto &radius : parameters.radii)
      {
        const Real r(radius);
        const Real f2 = evaluator.evaluate(r, s, sinc_step, use_split).rhs;
        AssertThrow(boost::math::isfinite(f2),
                    dealii::ExcMessage("The f_2 evaluation is not finite."));
        std::cout << "r = " << radius << ", f_2(r) = " << f2 << '\n';
      }
  }
} // namespace RSOF2

int
main(int argc, char **argv)
{
  try
    {
      const std::string usage =
        "Usage: evaluate_f2 [parameters.prm]\n"
        "       evaluate_f2 --write-parameters FILE\n"
        "Values of f_2 are printed to stdout.\n"
        "Timing and error messages are written to stderr.\n";
      RSOF2::Parameters parameters;
      if (argc == 2 && std::string(argv[1]) == "--help")
        {
          std::cout << usage;
          return 0;
        }
      if (argc == 3 && std::string(argv[1]) == "--write-parameters")
        {
          parameters.write_defaults(argv[2]);
          return 0;
        }
      AssertThrow(argc <= 2, dealii::ExcMessage(usage));
      const std::filesystem::path filename =
        argc == 2 ?
          std::filesystem::path(argv[1]) :
          std::filesystem::path(RSO_F2_SOURCE_DIR) / "parameters_2d.prm";
      parameters.parse(filename);
      // Set the process-wide GSL policy once, before any evaluations. Checked
      // GSL calls can then report failures through deal.II exceptions.
      gsl_set_error_handler_off();

      std::cout << "f_2: dimension = " << parameters.dimension
                << ", s = " << parameters.exponent
                << ", kappa = " << parameters.kappa
                << ", method = " << parameters.method
                << ", precision = " << parameters.precision_bits << " bits\n";

      dealii::TimerOutput timer(std::cerr,
                                dealii::TimerOutput::never,
                                dealii::TimerOutput::cpu_and_wall_times);
      {
        // This example timer includes printing the values to the console.
        dealii::TimerOutput::Scope scope(timer, "Evaluate and print f_2");
        if (parameters.precision_bits == 53)
          RSOF2::evaluate_double(parameters);
        else if (parameters.precision_bits == 160)
          RSOF2::evaluate_multiprecision<RSOF2::Number<160>>(parameters);
        else if (parameters.precision_bits == 224)
          RSOF2::evaluate_multiprecision<RSOF2::Number<224>>(parameters);
        else
          RSOF2::evaluate_multiprecision<RSOF2::Number<256>>(parameters);
      }
      std::cout.flush();
      AssertThrow(std::cout, dealii::ExcIO());
      if (parameters.print_timing)
        timer.print_summary();
      return 0;
    }
  catch (const std::exception &error)
    {
      std::cerr << "f_2 evaluation: " << error.what() << '\n';
      return 1;
    }
}
