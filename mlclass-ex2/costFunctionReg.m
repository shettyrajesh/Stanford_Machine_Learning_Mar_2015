function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


theta_1 = theta(2:size(theta(:, 1)), 1);
hx = sigmoid(X * theta);
J = 1/m * sum(-y' * log(hx) - (1-y)' * log(1-hx)) + lambda / (2 * m) * sum(theta_1 .^ 2);


for i = 1:size(theta(:,1))
    X_i = X(:,i);
    if (i == 1)
        grad(i,1) = 1/m * sum((hx-y) .* X_i);
    else
        grad(i,1) = (1/m * sum((hx-y) .* X_i)) + lambda/m * theta(i,1);
    endif  

%J=(-1.0/m) * sum(y.*log(sigmoid(X*theta)) + (1-y).*log(1-sigmoid(X.theta))) + (lambda*1.0/2*m) * (sum(theta.**2) - theta(1).^2);

%grad=(1.0/m)* X'*(sigmoid(X*theta)-y) + (lambda*1.0/m) * theta;
%grad(1)=(1.0/m) * (X(:,1))'*sigmoid(x*theta)-y);

%grad=(1.0/m)* sum(X'*(sigmoid(X*theta)-y)) + (lambda*1.0/m) * theta;
%grad(1)=(1.0/m) * sum((X(:,1))'*sigmoid(x*theta)-y));

%J=(-1.0/m) * sum(y.*log(sigmoid(X*theta)) + (1-y).*log(1-sigmoid(X.theta)));
%grad=(1.0/m)* X'*(sigmoid(X*theta)-y);

%temp = theta;
%temp(1)=0;
%grad = grad + (lambda*1.0/m) * temp
%J = J + (lambda*1.0/2*m) * sum(temp.**2);


% =============================================================

end
