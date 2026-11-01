#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/patterns.h>
#include <deal.II/base/utilities.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace RSOF2
{
  // Parameters for evaluating f_2, without any reference-data dependency.
  // Decimal inputs remain strings until converted to the chosen arithmetic.
  class Parameters
  {
  public:
    Parameters()
    {
      using namespace dealii;
      prm.enter_subsection("Evaluation");
      prm.declare_entry("Dimension", "2", Patterns::Selection("2|3"));
      prm.declare_entry("Fractional exponent", "0.5", Patterns::Double(0, 1));
      prm.declare_entry("Kappa", "1", Patterns::Double(0));
      prm.declare_entry("Radii",
                        "0, 0.5, 1, 1.5, 2",
                        Patterns::List(Patterns::Double(0), 1));
      prm.declare_entry("Method", "split", Patterns::Selection("direct|split"));
      prm.declare_entry(
        "Working precision bits",
        "160",
        Patterns::Selection("53|160|224|256"),
        "53 selects double; 160/224/256 select Boost multiprecision.");
      prm.leave_subsection();

      prm.enter_subsection("Quadrature");
      prm.declare_entry("Sinc spacing", "0.20", Patterns::Double(0));
      prm.declare_entry("Gaussian order",
                        "48",
                        Patterns::Integer(2, std::numeric_limits<int>::max()));
      prm.declare_entry(
        "Sinc truncation",
        "balanced",
        Patterns::Selection("balanced|tail tolerance"),
        "Choose balanced indices or an exponential-tail target.");
      prm.declare_entry("Tail tolerance",
                        "1e-12",
                        Patterns::Double(0, 1),
                        "Used only with tail tolerance truncation; this is not "
                        "a bound on the total evaluation error.");
      prm.leave_subsection();

      prm.enter_subsection("Output");
      prm.declare_entry("Print timing summary", "true", Patterns::Bool());
      prm.leave_subsection();
    }

    void
    parse(const std::filesystem::path &filename)
    {
      prm.parse_input(filename.string());
      prm.enter_subsection("Evaluation");
      dimension      = prm.get_integer("Dimension");
      exponent       = prm.get("Fractional exponent");
      kappa          = prm.get("Kappa");
      radii          = dealii::Utilities::split_string_list(prm.get("Radii"));
      method         = prm.get("Method");
      precision_bits = prm.get_integer("Working precision bits");
      prm.leave_subsection();
      prm.enter_subsection("Quadrature");
      step           = prm.get("Sinc spacing");
      order          = prm.get_integer("Gaussian order");
      truncation     = prm.get("Sinc truncation");
      tail_tolerance = prm.get("Tail tolerance");
      prm.leave_subsection();
      prm.enter_subsection("Output");
      print_timing = prm.get_bool("Print timing summary");
      prm.leave_subsection();

      const double s = finite_value(exponent, "Fractional exponent");
      AssertThrow(s > 0 && s < 1,
                  dealii::ExcMessage(
                    "The fractional exponent must lie in (0,1)."));
      AssertThrow(finite_value(kappa, "Kappa") > 0,
                  dealii::ExcMessage("Kappa must be positive."));
      AssertThrow(finite_value(step, "Sinc spacing") > 0,
                  dealii::ExcMessage("The sinc spacing must be positive."));
      for (const auto &r : radii)
        AssertThrow(finite_value(r, "Radius") >= 0,
                    dealii::ExcMessage("Radii must be nonnegative."));
      AssertThrow(dimension == 2 || method == "direct",
                  dealii::ExcMessage(
                    "The split method currently supports dimension 2 only."));
      const double tail = finite_value(tail_tolerance, "Tail tolerance");
      AssertThrow(tail > 0 && tail < 1,
                  dealii::ExcMessage("The tail tolerance must lie in (0,1)."));
    }

    void
    write_defaults(const std::filesystem::path &filename) const
    {
      std::ofstream out(filename);
      AssertThrow(out, dealii::ExcFileNotOpen(filename.string()));
      prm.print_parameters(out, dealii::ParameterHandler::PRM);
      out.flush();
      AssertThrow(out, dealii::ExcIO());
    }

    unsigned int dimension = 2, precision_bits = 160, order = 48;
    std::string  exponent = "0.5", kappa = "1", step = "0.20", method = "split";
    std::string  truncation = "balanced", tail_tolerance = "1e-12";
    std::vector<std::string> radii;
    bool                     print_timing = true;

  private:
    dealii::ParameterHandler prm;

    static double
    finite_value(const std::string &text, const std::string &name)
    {
      const double value = std::stod(text);
      AssertThrow(std::isfinite(value),
                  dealii::ExcMessage(name + " must be finite."));
      return value;
    }
  };
} // namespace RSOF2
