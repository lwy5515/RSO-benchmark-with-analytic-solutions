function example_f2()
% example_f2  Print the one-dimensional f_2 on [-2,2] as CSV.
%   The output columns are x,f2. No output files are created.
%   Han Xiao
%   Oct. 24, 2026

  x = -2:0.1:2;
  kappa = 1;
  s = 0.5;
  f = rhs_cosh2(x, kappa, s);

  fprintf('x,f2\n');
  fprintf('%.17g,%.17g\n', [x(:),f(:)].');
end
