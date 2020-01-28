% Author:- Shraavan Sivasubramanian
% Created on 30th November 2019

function [normal_images, smiling_images] = LoadDataSet(num_images, im_height, im_width)
% LOADDATASET - a function which loads all images (examples) in the dataset.
% All images are in a folder called 'image_set'. The filename format for
% each normal/smiling image, is a number followed by 'a'/'b' respectively.
% Input Args:-
  % num_images - the total number of images.
  % im_height - the height of each image (number of rows).
  % im_width - the width of each image (number of columns).
% Output Args:-
  % normal_images - a matrix of all the normal images.
  % smiling_images - a matrix of all the smiling images.
  
  cd image_set;
  normal_images = zeros(im_height*im_width, num_images);
  smiling_images = zeros(im_height*im_width, num_images);

  for i=1:num_images
    temp_img = imread(strcat(num2str(i),'a.jpg'));
    normal_images(:, i) = temp_img(:); 
    temp_img = imread(strcat(num2str(i),'b.jpg'));
    smiling_images(:, i) = temp_img(:);
  end
  cd ..;
end

