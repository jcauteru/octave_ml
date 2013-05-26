function [J grad] = nnCostFunction_vect(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


%layer one
X = [ones(m, 1) X];
%z_2 = a_1*Theta1';
out1 = sigmoid(X*Theta1');

%layer two
out2 = [ones(m, 1) out1];
%z_3 = a_2*Theta2';
h = sigmoid(out2*Theta2');

for i = 1:num_labels,
y_all(:,i) = (y == i);
end


big_delta_2=0;
big_delta_1=0;


delta_3_all = (h - y_all);
delta_2_all = (delta_3_all*Theta2).*(sigmoidGradient([ones(m, 1) X*Theta1']));



big_delta_1 = (X'*delta_2_all(:,2:end))';
big_delta_2 = (out2'*delta_3_all)';


Theta1_grad_unreg = big_delta_1/m;
Theta2_grad_unreg = big_delta_2/m;

reg_t_theta1 = (lambda*Theta1)/m;
reg_t_theta1(:, 1) = 0;

Theta1_grad = Theta1_grad_unreg + reg_t_theta1;


reg_t_theta2 = (lambda*Theta2)/m;
reg_t_theta2(:, 1) = 0;

Theta2_grad = Theta2_grad_unreg + reg_t_theta2;


%cost
h_log = log(h);
log_one_minus_h = log(1-h);


minus_y = y_all.*(-1);
one_minus_y = (1+minus_y);


J_noreg = (sum(sum((minus_y.*h_log)-(one_minus_y.*log_one_minus_h),2))/m);

Theta1_gawesome = sum(sum((Theta1(:,2:input_layer_size+1).^2),2));
Theta2_gawesome = sum(sum((Theta2(:,2:hidden_layer_size+1).^2),2));


J = J_noreg + ((lambda*(Theta1_gawesome+Theta2_gawesome))/(2*m));



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
