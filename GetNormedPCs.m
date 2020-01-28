% Author:- Shraavan Sivasubramanian
% Created on 30th November 2019

function [V] = GetNormedPCs(U)
% GETKPCS - a function which returns K Principal Components, with unit
% norm.
% Input Args:-
  % U - the matrix of all eigenvectors.
% Output Args:-
  % V - the matrix of eigenvectors with unit norm.

V = zeros(size(U));

for i=1:size(U, 2)
    V(:,i) = U(:,i)/sqrt(transpose(U(:,i))*U(:,i));
end

end

