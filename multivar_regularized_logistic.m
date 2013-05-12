function [J, grad] = costFunctionReg(theta, X, y, lambda)


% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));


z = X*theta;
h = sigmoid(z);
h_log = log(h);

sq_theta_sum = (lambda*(sum(theta(2:rows(theta)).^2)))/(2*m);

J_1 = sum(-y.*h_log-(1-y).*log(1-h))/m;

J = J_1 + sq_theta_sum;

theta_parama = zeros(size(theta));
theta_parama(2:length(theta)) = 1;

theta_mod = theta_parama.*((lambda/m).*theta);

grad = ((X'*(h-y))./m) + theta_mod;
%grad = 0;


end
