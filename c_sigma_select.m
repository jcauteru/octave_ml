function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%create c and sigma selection matrix
max_mult = 5;
seeds = [.0001 .0003]

c_vect = zeros(5, 2);

for m = 1:5,
	multi = 10^m;
	c_vect(m,:) = seeds*multi;
end;

all_poss = c_vect(:)

rows_comb = 0;
for i = 1:rows(all_poss),
for j = 1:rows(all_poss),
my_combos(rows_comb + j, 1) = all_poss(i);
my_combos(rows_comb + j, 2) = all_poss(j);
end;
rows_comb = rows(my_combos);	
end;


for i = 1:rows(my_combos),
C_try = my_combos(i,1);
sigma_try = my_combos(i, 2);

model= svmTrain(X, y, C_try, @(x1, x2) gaussianKernel(x1, x2, sigma_try));
pred = svmPredict(model, Xval);

error(i) = mean(double(pred ~= yval));
end;

index = find(error == min(error)) 

C = my_combos(index,1);
sigma = my_combos(index,2);


% =========================================================================

end
