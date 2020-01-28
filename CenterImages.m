% Author:- Shraavan Sivasubramanian
% Created on 30th November 2019

function [centered_images, mean_image] = CenterImages(images)
% CENTERIMAGES - a function which subtracts the mean image from all the
% images.
% It assumes that each column of the matrix 'images' is an example image.
% So the mean is taken across all columns.
% Input Args:-
  % images - the matrix of example images.
% Output Args:-
  % centered_images - the result after subtracting the mean image.
  % mean_image - the mean of all images.

mean_image = mean(images, 2);
centered_images = images - mean_image;

end

