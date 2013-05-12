function [theta, J_history] = gradientDescentMulti2(X, y, theta, alpha, num_iters)

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
	    
	
	inside= (1/m)*((X*theta)-y);
	theta = theta -  (alpha*(inside'*X)');
    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
