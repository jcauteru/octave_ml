function [theta] = normalEqn(X, y)

theta = zeros(size(X, 2), 1);

theta = ((X'*X)^-1)*X'*y

end