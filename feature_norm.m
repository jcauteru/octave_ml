function [X_norm, mu, sigma] = featureNormalize(X)

X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));
     
for i = 1:columns(X),
	hold_x = X(:,i);
	mu = mean(hold_x);
	summed = sum((hold_x - mu).^2);
	sigma(:,i) = sqrt(summed/(rows(hold_x)-1));
	sigma2 = (hold_x-mu)./sigma(:,i);
	X_norm(:,i)=sigma2;
end
	
end
