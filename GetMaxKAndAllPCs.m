% Author:- Shraavan Sivasubramanian
% Created on 30th November 2019

function [U,S,K] = GetMaxKAndAllPCs(M, frac_var_ret, decomp_str)
% GETMAXKANDALLPCs - a function which computes the principal components.
% To determine how many components to take, we will compare the partial sum
% of the singular values to the sum of all singular values. The ratio of
% these 2 sums is a measure of the amount of variance that is retained,
% since the diagonal entries of the covariance matrix is the variance along
% that dimension.
% Input Args:-
  % M - the input matrix for which the principal components need to be
  % calculated.
  % frac_var_ret - the least fraction of total variance that needs to be
  % retained. It can take on values between 0 and 1.
  % decomp_str - a string denoting the type of decomposition.
% Output Args:-
  % U - the matrix of eigenvectors/left singular vectors, each vector being
  % a column of U (if SVD is used, they have unit norm).
  % S - the diagonal matrix of eigenvalues/singular values.
  % K - the maximum number of principal components to consider, based on
  % frac_var_ret.

if strcmpi(decomp_str, "eigen")==1
    % There's no guarantee that the returned eigenvalues are sorted in any 
    % order. So as a safety measure, I sort the eigen values in descending 
    % order.
    [U,S] = eig(M);
    [~,ind] = sort(diag(S), 'descend');
    S = S(ind, ind);
    U = U(:, ind);
elseif strcmpi(decomp_str, "svd")==1
    % No need to sort the singular values as svd() returns them in
    % descending order.
    [U,S] = svd(M);
else
    error("Invalid Decomposition Argument passed.");
end

sing_vals = diag(S);
sum_all_sing_val = sum(sing_vals);

for i=1:length(sing_vals)
    temp_sum = sum(sing_vals(1:i));
    retained_var = temp_sum/sum_all_sing_val;
    if retained_var >= frac_var_ret
        break;
    end
end

K = i;

end

