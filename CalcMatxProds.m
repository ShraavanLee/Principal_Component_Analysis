% Author:- Shraavan Sivasubramanian
% Created on 30th November 2019

function [ATA,AAT] = CalcMatxProds(A)
%CALCMATXPRODS - a function which computes (for a given matrix A) the
%products A^T*A and A*A^T.
% Input Args:-
  % A - the input matrix.
% Output Args:-
  % ATA - the product A^T*A.
  % AAT - the product A*A^T.

num_imgs = size(A,2);
ATA = transpose(A)*A/num_imgs;
AAT = A*transpose(A)/num_imgs;

end

