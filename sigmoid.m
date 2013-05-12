function g = sigmoid(z)

% You need to return the following variables correctly 
g = zeros(size(z));

min_z = z.*(-1);

g = 1./(1.+(e.^min_z));

end
