function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    
    theta

	inside_sum = ((X*theta) - y);
	
	x_1_sum = sum(inside_sum.*X(:,1));
	
	x_2_sum = sum(inside_sum.*X(:,2));
	
	theta_0 = theta(1) - alpha*(1/m)*x_1_sum;
	
	theta_1 = theta(2) - alpha*(1/m)*x_2_sum;	
	
	theta = [theta_0; theta_1];
  
    J_history(iter) = computeCost(X, y, theta);

end

end
