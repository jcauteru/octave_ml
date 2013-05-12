function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

p = zeros(m, 1);

%options = optimset('GradObj', 'on', 'MaxIter', 400);
%[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

p_2 = X*theta;

for i = 1:rows(p),
	if (p_2(i) >=0) 
	p(i) = 1; 
	endif
	
end


end
