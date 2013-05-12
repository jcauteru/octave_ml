function J = computeCostMulti(X, y, theta)

% Initialize some useful values
m = length(y); % number of training examples

J = 0;


%take x*theta1
h = (X*theta);

%produce errors
errorsq = (h-y).^2;

J = 1/(2*m) * sum(errorsq);

% =========================================================================

end
