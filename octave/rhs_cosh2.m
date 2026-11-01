function f = rhs_cosh2(x, kappa, s)
% RHS_COSH2  The explicit f_2 formula.
%   f = rhs_cosh2(x,kappa,s) evaluates (kappa^2-Delta)^s u_2, where
%   u_2(x) = (1-x^2)^2 cosh(kappa*x) for |x|<1 and is zero outside.
%   Here kappa>0, 0<s<1; x may be a real scalar, vector or array.
%   The output has the same size as x and includes exterior RHS values.
%
%   Han Xiao
%   Oct. 24, 2026

  validateattributes(x, {'numeric'}, {'real','finite'}, mfilename, 'x');
  validateattributes(kappa, {'numeric'}, ...
                     {'real','finite','scalar','positive'}, mfilename, 'kappa');
  validateattributes(s, {'numeric'}, ...
                     {'real','finite','scalar','>',0,'<',1}, mfilename, 's');
                     
  x = double(x); 
  kappa = double(kappa); 
  s = double(s);
  
  sh = sinh(kappa); 
  ch = cosh(kappa);

  c = [384*kappa^5*sh; ...
       -96*kappa^3*(2*kappa*ch + 5*sh); ...
       8*kappa*(24*kappa*ch + (4*kappa^2 + 15)*sh); ...
       -24*(ch + kappa*sh)];
  d = [192*kappa^4*ch; ...
       -48*kappa^2*(3*ch + 2*kappa*sh); ...
       4*((4*kappa^2 + 3)*ch + 12*kappa*sh); ...
       -4*ch];

  f = zeros(size(x));
  for j = 0:3
    alpha = 5-s-j;
    f = f + c(j+1)*(G_gamma(alpha,x+1,kappa) ...
                      + G_gamma(alpha,x-1,kappa));
  end
  for j = 0:3
    alpha = 4-s-j;
    f = f + d(j+1)/alpha*(yG_gamma(alpha,x+1,kappa) ...
                           - yG_gamma(alpha,x-1,kappa));
  end
end

function value = G_gamma(alpha, y, kappa)
  r = abs(y); nu = alpha-0.5;
  value = zeros(size(y));
  nonzero = (r>0);
  value(nonzero) = r(nonzero).^nu .* besselk(nu,kappa*r(nonzero)) ...
                  / (sqrt(pi)*gamma(alpha)*(2*kappa)^nu);
  % G_\alpha(0)
  if any(~nonzero(:))
    if alpha>0.5
      value(~nonzero) = gamma(alpha-0.5) ...
                       / (2*sqrt(pi)*gamma(alpha)*kappa^(2*alpha-1));
    else
      value(~nonzero) = Inf;
    end
  end
end

function value = yG_gamma(alpha, y, kappa)
  value = zeros(size(y));
  nonzero = (y~=0);
  value(nonzero) = y(nonzero).*G_gamma(alpha,y(nonzero),kappa);
end
