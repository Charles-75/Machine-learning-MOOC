function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
h = X * theta;
J1 = (1/(2*m))* sum((h-y).^2); 
grad1=(1/m)*((X'*(h-y)));
theta(1)=0;
regularization = (lambda/(2*m))*sum(theta.^2);
J=J1+regularization;
grad_reg=(lambda/m)*theta;
grad=grad1+grad_reg;
%              
%












% =========================================================================

grad = grad(:);

end
