% Author:- Shraavan Sivasubramanian
% Created on 30th November 2019

%% Face Recognition using Principal Component Analysis.

clear;
clc;


%% Read Images. Normal images and Smiling images in separate matrices.
% The matrix of images is in the foll. format:- [features x num_examples],
% where features = im_height*im_width. So every column is an example img.

num_images = 200;
im_height = 193;
im_width = 162;

[normal_images, smiling_images] = LoadDataSet(num_images, im_height, im_width);

%% Calculate the mean face and subtract it from the examples.
% We use only the first 190 images.

[normal_img_centered, normal_img_mean] = CenterImages(normal_images(:, 1:190));
[smiling_img_centered, smiling_img_mean] = CenterImages(smiling_images(:, 1:190));

%% Calculate the covariance matrix (AA') for the normal images, and also the matrix A'A.
% Here, the matrix A is denoted using the variable 'normal_img_centered',
% AA' is denoted using 'cov_mat_normal' and A'A is denoted using 'M'.

[M, cov_mat_normal] = CalcMatxProds(normal_img_centered);

%% Determining Principal Components using eigen decomposition/SVD

fract_var_ret = 0.95;

[U,S,K] = GetMaxKAndAllPCs(M, fract_var_ret, "eigen");

%% Part a)
% Compute the principal components (PCs) using first 190 individuals’ neutral expression
% image. Plot the singular values of the data matrix and justify your choice of
% principal components.

figure(1);
set(gcf, 'Position', get(0,'Screensize'));
plot(diag(S), 'LineWidth', 2);
xlabel('Index of Principal Components (PCs)');
ylabel('Singular value at Index');
title('Singular Values of A^{T}A; A - matrix of centered normal images');
saveas(gcf, 'part_a', 'jpeg');

%% Normalize the eigenvectors and select the top K based on eigenvalues/singular values.

eig_vec = normal_img_centered*U;
eig_vec = GetNormedPCs(eig_vec);
eig_vec = eig_vec(:, 1:K);

%% Part b)
% Reconstruct one of 190 individuals’ neutral expression image using different number
% of PCs. As you vary the number of PCs, plot the mean squared error (MSE) of
% reconstruction versus the number of principal components to show the accuracy of
% reconstruction. Comment on your result.

idx = randi([1, 190]);
img_to_reconstruct = normal_img_centered(:, idx);
figure(2);
set(gcf, 'Position', get(0,'Screensize'));
imagesc(reshape(normal_images(:, idx), im_height, im_width));
colormap gray;
title('Original Image');
saveas(gcf, 'part_b_1', 'jpeg');

num_PCs = ceil(linspace(1, K, 10));
num_imgs_to_disp = 4; % display images for 4 choices of PCs
ind_to_disp = randsample(num_PCs, num_imgs_to_disp); 
ind_to_disp = sort(ind_to_disp);
disp_images = zeros(im_height*im_width, num_imgs_to_disp);
disp_mse = zeros(num_imgs_to_disp, 1); % for getting corresponding MSEs.
i = 1; % to index over disp_images.

normal_mse = [];
for k=num_PCs
   reconstructed_img = zeros(im_height*im_width, 1);
   for j=1:k
       w_j = transpose(eig_vec(:,j))*img_to_reconstruct;
       reconstructed_img = reconstructed_img + w_j*eig_vec(:,j);
   end
   reconstructed_img = reconstructed_img + normal_img_mean;
   err = normal_images(:, idx) - reconstructed_img;
   normal_mse = [normal_mse, transpose(err)*err/length(err)];
   if ~isempty(find(ind_to_disp == k))
       disp_images(:,i) = reconstructed_img;
       disp_mse(i) = transpose(err)*err/length(err);
       i = i + 1;
   end
end

figure(3);
set(gcf, 'Position', get(0,'Screensize'));
subplot(2,2,1);
imagesc(reshape(disp_images(:,1), im_height, im_width));
colormap gray;
title(['Image Reconstructed with ', num2str(ind_to_disp(1)), ' PCs']);
xlabel(['MSE = ', num2str(disp_mse(1))]);
subplot(2,2,2);
imagesc(reshape(disp_images(:,2), im_height, im_width));
colormap gray;
title(['Image Reconstructed with ', num2str(ind_to_disp(2)), ' PCs']);
xlabel(['MSE = ', num2str(disp_mse(2))]);
subplot(2,2,3);
imagesc(reshape(disp_images(:,3), im_height, im_width));
colormap gray;
title(['Image Reconstructed with ', num2str(ind_to_disp(3)), ' PCs']);
xlabel(['MSE = ', num2str(disp_mse(3))]);
subplot(2,2,4);
imagesc(reshape(disp_images(:,4), im_height, im_width));
colormap gray;
title(['Image Reconstructed with ', num2str(ind_to_disp(4)), ' PCs']);
xlabel(['MSE = ', num2str(disp_mse(4))]);
saveas(gcf, 'part_b_2', 'jpeg');

figure(4);
set(gcf, 'Position', get(0,'Screensize'));
plot(num_PCs, normal_mse, 'b', 'Marker', '+', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('Number of PCs used in Reconstruction');
ylabel('MSE');
title(['Effect of Number of PCs used in Reconstructing the ', num2str(idx),...
    '^{th} Normal Face Image']);
saveas(gcf, 'part_b_3', 'jpeg');

%% Part c)
% Reconstruct one of 190 individuals’ smiling expression image using different number
% of PCs. Again, plot the MSE of reconstruction versus the number of principal
% components and comment on your result.

% Note:- for the sake of comparison, I will use the same index, so that the
% corresponding smiling image can be compared. We could pick an index at
% random and do the reconstruction as well, in which case, the
% corresponding code should also be copied from the previous section here.

img_to_reconstruct = smiling_img_centered(:, idx);
figure(5);
set(gcf, 'Position', get(0,'Screensize'));
imagesc(reshape(smiling_images(:, idx), im_height, im_width));
colormap gray;
title('Original Image');
saveas(gcf, 'part_c_1', 'jpeg');

num_imgs_to_disp = 4; % display images for 4 choices of PCs
ind_to_disp = randsample(num_PCs, num_imgs_to_disp); 
ind_to_disp = sort(ind_to_disp);
disp_images = zeros(im_height*im_width, num_imgs_to_disp);
disp_mse = zeros(num_imgs_to_disp, 1); % for getting corresponding MSEs.
i = 1; % to index over disp_images.

smiling_mse = [];
for k=num_PCs
   reconstructed_img = zeros(im_height*im_width, 1);
   for j=1:k
       w_j = transpose(eig_vec(:,j))*img_to_reconstruct;
       reconstructed_img = reconstructed_img + w_j*eig_vec(:,j);
   end
   reconstructed_img = reconstructed_img + smiling_img_mean;
   err = smiling_images(:, idx) - reconstructed_img;
   smiling_mse = [smiling_mse, transpose(err)*err/length(err)];
   if ~isempty(find(ind_to_disp == k))
       disp_images(:,i) = reconstructed_img;
       disp_mse(i) = transpose(err)*err/length(err);
       i = i + 1;
   end
end

figure(6);
set(gcf, 'Position', get(0,'Screensize'));
subplot(2,2,1);
imagesc(reshape(disp_images(:,1), im_height, im_width));
colormap gray;
title(['Image Reconstructed with ', num2str(ind_to_disp(1)), ' PCs']);
xlabel(['MSE = ', num2str(disp_mse(1))]);
subplot(2,2,2);
imagesc(reshape(disp_images(:,2), im_height, im_width));
colormap gray;
title(['Image Reconstructed with ', num2str(ind_to_disp(2)), ' PCs']);
xlabel(['MSE = ', num2str(disp_mse(2))]);
subplot(2,2,3);
imagesc(reshape(disp_images(:,3), im_height, im_width));
colormap gray;
title(['Image Reconstructed with ', num2str(ind_to_disp(3)), ' PCs']);
xlabel(['MSE = ', num2str(disp_mse(3))]);
subplot(2,2,4);
imagesc(reshape(disp_images(:,4), im_height, im_width));
colormap gray;
title(['Image Reconstructed with ', num2str(ind_to_disp(4)), ' PCs']);
xlabel(['MSE = ', num2str(disp_mse(4))]);
saveas(gcf, 'part_c_2', 'jpeg');

figure(7);
set(gcf, 'Position', get(0,'Screensize'));
plot(num_PCs, normal_mse, 'b', 'Marker', '+', 'MarkerSize', 10, 'LineWidth', 2);
hold on;
plot(num_PCs, smiling_mse, 'r', 'Marker', 'o', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('Number of PCs used in Reconstruction');
ylabel('MSE');
title(['Effect of Number of PCs used in Reconstructing the ', num2str(idx), '^{th} Image']);
legend([num2str(idx), '^{th} Normal Image'], [num2str(idx), '^{th} Smiling Image']);
saveas(gcf, 'part_c_3', 'jpeg');

%% Part d)
% Reconstruct one of the other 10 individuals’ neutral expression image using different
% number of PCs. Again, plot the MSE of reconstruction versus the number of principal
% components and comment on your result.

% This time, we don't have the centered image for images 191-200. So use
% the same mean as the first 190 images.

test_idx = randi([191, num_images]);
img_to_reconstruct = normal_images(:, test_idx) - normal_img_mean;
figure(8);
set(gcf, 'Position', get(0,'Screensize'));
imagesc(reshape(normal_images(:, test_idx), im_height, im_width));
colormap gray;
title('Original Image');
saveas(gcf, 'part_d_1', 'jpeg');

num_imgs_to_disp = 4; % display images for 4 choices of PCs
ind_to_disp = randsample(num_PCs, num_imgs_to_disp); 
ind_to_disp = sort(ind_to_disp);
disp_images = zeros(im_height*im_width, num_imgs_to_disp);
disp_mse = zeros(num_imgs_to_disp, 1); % for getting corresponding MSEs.
i = 1; % to index over disp_images.

test_mse = [];
for k=num_PCs
   reconstructed_img = zeros(im_height*im_width, 1);
   for j=1:k
       w_j = transpose(eig_vec(:,j))*img_to_reconstruct;
       reconstructed_img = reconstructed_img + w_j*eig_vec(:,j);
   end
   reconstructed_img = reconstructed_img + normal_img_mean;
   err = normal_images(:, test_idx) - reconstructed_img;
   test_mse = [test_mse, transpose(err)*err/length(err)];
   if ~isempty(find(ind_to_disp == k))
       disp_images(:,i) = reconstructed_img;
       disp_mse(i) = transpose(err)*err/length(err);
       i = i + 1;
   end
end

figure(9);
set(gcf, 'Position', get(0,'Screensize'));
subplot(2,2,1);
imagesc(reshape(disp_images(:,1), im_height, im_width));
colormap gray;
title(['Image Reconstructed with ', num2str(ind_to_disp(1)), ' PCs']);
xlabel(['MSE = ', num2str(disp_mse(1))]);
subplot(2,2,2);
imagesc(reshape(disp_images(:,2), im_height, im_width));
colormap gray;
title(['Image Reconstructed with ', num2str(ind_to_disp(2)), ' PCs']);
xlabel(['MSE = ', num2str(disp_mse(2))]);
subplot(2,2,3);
imagesc(reshape(disp_images(:,3), im_height, im_width));
colormap gray;
title(['Image Reconstructed with ', num2str(ind_to_disp(3)), ' PCs']);
xlabel(['MSE = ', num2str(disp_mse(3))]);
subplot(2,2,4);
imagesc(reshape(disp_images(:,4), im_height, im_width));
colormap gray;
title(['Image Reconstructed with ', num2str(ind_to_disp(4)), ' PCs']);
xlabel(['MSE = ', num2str(disp_mse(4))]);
saveas(gcf, 'part_d_2', 'jpeg');

figure(10);
set(gcf, 'Position', get(0,'Screensize'));
plot(num_PCs, test_mse, 'r', 'Marker', 'd', 'MarkerSize', 10, 'LineWidth', 2);
hold on;
plot(num_PCs, normal_mse, 'b', 'Marker', '+', 'MarkerSize', 10, 'LineWidth', 2);
hold on;
xlabel('Number of PCs used in Reconstruction');
ylabel('MSE');
title('Effect of Number of PCs used in Reconstructing Various Face Images');
legend([num2str(test_idx), '^{th} Normal Image'], [num2str(idx), '^{th} Normal Image']);
saveas(gcf, 'part_d_3', 'jpeg');

%% Part e)
% Use any other non-human image (e.g., car image, resize and crop to the same size),
% and try to reconstruct it using all the PCs. Comment on your results.

% We will use the same mean (normal_img_mean) as earlier.

% Read the image.
non_human_img = imread('rose.jpeg');

% Convert from RGB to Grayscale. Sometimes grayscale images also get read
% as an x*y*z matrix.
non_human_img = rgb2gray(non_human_img);

% Resize the image according to the height and width.
non_human_img = double(imresize(non_human_img, [im_height im_width]));

non_human_mse = [];
img_to_reconstruct = non_human_img(:) - normal_img_mean;

for k=num_PCs
    reconstructed_img = zeros(im_height*im_width, 1);
    for j=1:k
        w_j = transpose(eig_vec(:,j))*img_to_reconstruct;
        reconstructed_img = reconstructed_img + w_j*eig_vec(:,j);
    end
    reconstructed_img = reconstructed_img + normal_img_mean;
    err = normal_images(:, test_idx) - reconstructed_img;
    non_human_mse = [non_human_mse, transpose(err)*err/length(err)];
end

figure(11);
set(gcf, 'Position', get(0,'Screensize'));
subplot(1,2,1);
imagesc(non_human_img);
colormap gray;
title('Original Image');
subplot(1,2,2);
imagesc(reshape(reconstructed_img, im_height, im_width));
colormap gray;
title(['Non-human Image Reconstructed with ', num2str(K), ' PCs']);
xlabel(['MSE = ', num2str(non_human_mse(end))]);
saveas(gcf, 'part_e_1', 'jpeg');

figure(12);
set(gcf, 'Position', get(0,'Screensize'));
plot(num_PCs, non_human_mse, 'b', 'Marker', '+', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('Number of PCs used in Reconstruction');
ylabel('MSE');
title('Effect of Number of PCs used in Reconstructing a Non-Human Image');
saveas(gcf, 'part_e_2', 'jpeg');

%% Part f)

% Rotate one of 190 individuals’ neutral expression image with different degrees and
% try to reconstruct it using all PCs. Comment on your results.

% Again for the sake of comparison, I use the same image in part b.
% All K PCs will be used for different angles of rotation.
% Bilinear interpolation is used, and bounding box is set to 'crop' so as
% to preserve image size.

num_angles = 12;
rot_angle = linspace(30,360,num_angles);
img_to_rotate = reshape(normal_images(:, idx), im_height, im_width);
sample_rotated_img = zeros(im_height, im_width); % to show a sample rotated image.
rot_mse = [];
i = 1; % to index over subplots for rotation.
figure(13);
set(gcf, 'Position', get(0,'Screensize'));
sgtitle(['Image Reconstruction After Rotating By Various Angles using ',...
    num2str(K), ' PCs']);
for theta=rot_angle
    rotate_image = imrotate(img_to_rotate, theta, 'crop');
    if theta == 30
        sample_rotated_img = rotate_image;
    end
    img_to_reconstruct = rotate_image(:);
    reconstructed_img = zeros(im_height*im_width, 1);
    for j=1:K
        w_j = transpose(eig_vec(:,j))*img_to_reconstruct;
        reconstructed_img = reconstructed_img + w_j*eig_vec(:,j);
    end
    reconstructed_img = reconstructed_img + normal_img_mean;
    err = rotate_image(:) - reconstructed_img;
    rot_mse = [rot_mse, transpose(err)*err/length(err)];
    subplot(2, num_angles/2, i);
    imagesc(reshape(reconstructed_img, im_height, im_width));
    colormap gray;
    title(['\theta = ', num2str(theta), ' ^{O}']);
    xlabel(['MSE = ', num2str(transpose(err)*err/length(err))]);
    i = i + 1;
end
saveas(gcf, 'part_f_1', 'jpeg');

figure(14);
set(gcf, 'Position', get(0,'Screensize'));
subplot(1,2,1);
imagesc(img_to_rotate);
colormap gray;
title('Original Image');
subplot(1,2,2);
imagesc(sample_rotated_img);
colormap gray;
title('Original Image Rotated by 30 Degrees');
saveas(gcf, 'part_f_2', 'jpeg');

figure(15);
set(gcf, 'Position', get(0,'Screensize'));
plot(rot_angle, rot_mse, 'b', 'Marker', '+', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('Counterclockwise Angle of Rotation (degrees)');
ylabel('MSE');
title(['Effect of Angle of Rotation When Reconstructing a Normal Face Image With ',...
    num2str(K), ' PCs']);
saveas(gcf, 'part_f_3', 'jpeg');

close all;