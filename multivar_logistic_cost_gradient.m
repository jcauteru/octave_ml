function [J, grad] = costFunction(theta, X, y)

% Initialize some useful values
m = length(y); % number of training examples
 
J = 0;
grad = zeros(size(theta));


z = X*theta;
h = sigmoid(z);
h_log = log(h);

J = sum(-y.*h_log-(1-y).*log(1-h))/m;

grad = (X'*(h-y))./m;





% =============================================================

end
