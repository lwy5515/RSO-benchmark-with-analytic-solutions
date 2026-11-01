# Numerical valuation of the right-hand side $f = (\kappa^2 - \Delta)^s u$

Numerical implementations of the right-hand side
$f_2 = (\kappa^2 - \Delta)^s u_2$, where

- **in 1D:** $u_2(x)=(1-x^2)^2\cosh(\kappa x)$ for $|x|<1$,
  with zero exterior values;
- **in 2D and 3D:** $u_2(x)=(1-|x|^2)^2$ for $|x|<1$,
  with zero exterior values. The right-hand side is radial, with $r=|x|$.

The routines evaluate interior, boundary and exterior RHS values.

## Usage

### Octave code for 1D

The function [rhs_cosh2.m](octave/rhs_cosh2.m) evaluates $f_2(x)$ for
$\kappa > 0$ and $0 < s < 1$. It accepts a scalar or an array and returns
values of the same size, using double precision.

```matlab
addpath('octave');
x = [-2, -1, -0.5, 0, 0.5, 1, 2];
kappa = 1;
s = 0.5;
f = rhs_cosh2(x, kappa, s);
disp([x(:), f(:)]);
```

To run the supplied example, which prints `x,f2` on $[-2,2]$:

```sh
octave --quiet --eval "addpath('octave'); example_f2"
```

### deal.II C++ code for 2D/3D

The C++ code requires a C++17 compiler, CMake >= 3.16, deal.II >= 9.4,
GSL, and Boost headers. Build and run the examples with:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DDEAL_II_DIR=/path/to/deal.II
cmake --build build -j2

./build/evaluate_f2 cpp/parameters_2d.prm
./build/evaluate_f2 cpp/parameters_3d.prm
```

Omit `-DDEAL_II_DIR` if CMake can locate deal.II automatically. If needed,
add `-DGSL_ROOT_DIR=/path/to/gsl` to the configuration command.

Edit [parameters_2d.prm](cpp/parameters_2d.prm) or
[parameters_3d.prm](cpp/parameters_3d.prm) to choose the radii $r$, $s$, $\kappa$,
and quadrature settings (see the reference for more details):

- `Method`: `direct` works in 2D and 3D; `split` is available in 2D.
- `Working precision bits`: `53` selects double precision;
  `160`, `224`, or `256` selects multiprecision.
- `Sinc spacing` and `Gaussian order` specify $l$ and $m$ in the paper.
- `Sinc truncation`: `balanced` uses the paper's balanced indices;
  `tail tolerance` uses the specified `Tail tolerance` to select the indices.
  This tolerance does not bound the total evaluation error.

The program prints `r = ..., f_2(r) = ...` to the console. Timing is printed
separately; set `Print timing summary = false` to omit it. Running without
a parameter file uses the 2D example. Use `--help` for command-line options.

For evaluation inside another program, use [rhs_evaluator.h](cpp/rhs_evaluator.h)
for double precision or
[rhs_evaluator_multiprecision.h](cpp/rhs_evaluator_multiprecision.h)
for multiprecision. The essential double-precision call is:

```cpp

gsl_set_error_handler_off();

RSOStudy::KernelParameters kernel;
const RSOStudy::Evaluator evaluator(/*m=*/48, kernel, /*d=*/2, /*kappa=*/1.0);
const double f2 = evaluator.evaluate(/*r=*/0.5, /*s=*/0.5,
                                     /*l=*/0.20, /*split=*/true).rhs;
std::cout << f2 << '\n';
```

See [evaluate_f2.cc](cpp/evaluate_f2.cc) for complete examples of both
precisions.

## Reference

Han Xiao, Mengxue Bai, and Wenyu Lei,
*Analytical and semi-analytical benchmarks for the simulation of the
pseudo-relativistic Schrödinger operator*.
