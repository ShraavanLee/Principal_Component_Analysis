# Principal_Component_Analysis
PCA Implementation in MATLAB

The code contains an implementation of principal component analysis presented in the paper "Turk, Matthew A., and Alex P. Pentland. "Face recognition using eigenfaces." Proceedings. 1991 IEEE Computer
Society Conference on Computer Vision and Pattern Recognition. IEEE, 1991".

The image dataset(s) can be downloaded from
1) http://fei.edu.br/~cet/frontalimages_spatiallynormalized_cropped_equalized_part1.zip
2) http://fei.edu.br/~cet/frontalimages_spatiallynormalized_cropped_equalized_part1.zip

The code has been segmented into the following files:-
1) PCA_main - the main file used to run PCA.
2) LoadDataSet - a function which loads the image dataset.
3) CenterImages - a function which centers images about the mean image.
4) CalcMatxProds - a function which calculates two matrix products given a single input.
5) GetMaxKAndAllPCs - a function which computes the principal components
6) GetNormedPCs - a function which returns principal components with unit norm.

Comments have been added in every file to ensure easy understanding of the code and concept.

To use the code, place all images in the dataset in a folder named 'image_set' in the current working directory.

The non-human image (part e in main) can be placed in the current working directory itself.
