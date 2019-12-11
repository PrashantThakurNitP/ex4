function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%printing below to just check their size
nn_params
 input_layer_size
                                   hidden_layer_size
                                   num_labels
                                   X
                                   y
                                   lambda

Theta1
Theta2


% -------------------------------------------------------------
I = eye(num_labels);
Y = zeros(m, num_labels);
for i = 1:m
  Y(i, :) = I(y(i), :);
end
y=Y;
X1=ones(size(X,1),1);
X2=[X1,X];
a1=X2*(Theta1');
a2=sigmoid(a1);%a2
a11=ones(size(a2,1),1);
a21=[a11,a2];
h1=sigmoid(a21*(Theta2'))
%predict1=sigmoid(predict1)

%[t,predict]=max(predict1,[],2)

m

%J2=sum((y.*log(predict))+((1.-y).*log(1.-predict)))
J2=((y).*log(h1))+((1.-y).*log(1.-h1));%elementswise product of predict and output matrix
%here taking transpose and multipling y and log(h1) donot working
%here we need to perform elements wise products
J1=-(1/m)*sum(sum(J2,2));
ttheta1=zeros(size(Theta1));
ttheta2=zeros(size(Theta2));
%here i have simply copied square of Theta1 and Theta2 in thetha1 and ttheta2 but 1st column of each by 0
%in this way 1st column of each row is not included
ttheta1(:,2:size(Theta1,2))=Theta1(:,2:size(Theta1,2)).^2;
ttheta2(:,2:size(Theta2,2))=Theta2(:,2:size(Theta2,2)).^2;
J3=(lambda/(2*m))*(sum(sum(ttheta1,2))+sum(sum(ttheta2,2)));
%J4=sum(J3)
J=J1+J3;
theta1=Theta1(:,2:size(Theta1,2))
theta2=Theta2(:,2:size(Theta2,2))
%grad calculation

a1=X2;  %putting oiginal names as per standard convention
a2=a21

delta3=h1-y;
delta2=(delta3)*(theta2).*sigmoidGradient(X2*(Theta1'));  %ignore bias unit !important donot use Theta1 here

Delta1=zeros(size(Theta1));
Delta2=zeros(size(Theta2));

%here no need of for loop from 1 to m as vectorised method automatically add for all test cases
  Delta1 = Delta1+(delta2')*(a1);
  Delta2 = Delta2+(delta3')*(a2);
%earlier i have put it in for loop but it is not required
  theta1 = [zeros(size(theta1,1),1),theta1];
  theta2 = [zeros(size(theta2,1),1),theta2];
  
  Theta1_grad = Delta1/m + lambda / m * theta1;%earlier i have put 1/m only on 1st term but it should be on both term
  Theta2_grad = Delta2/m + lambda / m * theta2;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
