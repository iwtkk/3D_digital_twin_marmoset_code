% ct_body_segmentation.m
%
% Step 1 of building the 3D marmoset body model: turn the whole-body micro-CT
% volume into a binary body label.
%
% The animal was scanned lying inside a cylindrical holder tube, and the scanner
% reconstructs a circular field of view inside a square matrix. Both the tube
% wall and the corners of the matrix are above 0 HU, so neither can be removed
% by intensity alone. They are removed geometrically instead: a single axial
% cross-section of each is extruded along z, because the scanner geometry and
% the tube do not move during the acquisition.
%
%   1. Crop      keep voxels inside the field of view, outside the holder tube,
%                above 0 HU, and within the axial range of the animal.
%   2. Clean     per axial slice, fill holes and drop small regions; then drop
%                short runs along x and along y.
%   3. Close     re-impose the CT support and fill the remaining holes.
%
% Output: a binary label (test_body2.nii.gz) on the same grid as the CT.
%
% Steps 2 and 3 of the pipeline -- surface smoothing and extraction of the
% triangular surface -- were performed in 3D Slicer. See README.md.
%
% Requires: Image Processing Toolbox (imfill, bwareaopen) and the
%           "Tools for NIfTI and ANALYZE image" toolbox
%           (load_untouch_nii, save_untouch_nii).
%
% This script supersedes kakou_corp.m and kakou_fill.m in matlab_original/,
% which it reproduces exactly.

%% ------------------------------------------------------------------ Inputs
ctFile     = '20221020_173232BFbBs003a003.nii';            % micro-CT, HU
fovFile    = 'body_sample.nii.gz';                         % field-of-view template
holderFile = 'tst2Segmentation-Segment_1-label.nii.gz';    % cylindrical holder tube
outFile    = 'test_body2.nii.gz';                          % binary body label

%% -------------------------------------------------------------- Parameters
fovSlice     = 675;    % axial slice of fovFile that defines the circular FOV
holderSlice  = 115;    % axial slice of holderFile that shows the tube

zFirst       = 84;     % first axial slice containing the animal
zLast        = 1255;   % last  axial slice containing the animal

minAreaAxial = 100;    % drop axial regions smaller than this (voxels)
minRun       = 10;     % drop runs along x / y shorter than this (voxels)
zCleanFirst  = 200;    % axial range over which the run removal is applied
zCleanLast   = 1100;

%% ------------------------------------------------------------------- Load
ct     = load_untouch_nii(ctFile);
fov    = load_untouch_nii(fovFile);
holder = load_untouch_nii(holderFile);

nx = size(ct.img, 1);
ny = size(ct.img, 2);
nz = size(ct.img, 3);

fovMask    = fov.img(:, :, fovSlice)       ~= 0;   % inside the reconstructed FOV
holderMask = holder.img(:, :, holderSlice) == 1;   % holder tube cross-section

%% --------------------------------------------------------------- 1. Crop
% Keep tissue only: inside the field of view, outside the tube, above 0 HU,
% and inside the axial range of the animal.
keep2d = fovMask & ~holderMask;

body = false(nx, ny, nz);
for k = zFirst:zLast
    body(:, :, k) = (ct.img(:, :, k) > 0) & keep2d;
end

%% --------------------------------------------------------------- 2. Clean
% Axial planes: imfill closes the air-filled lumina (lung, trachea, gut) so
% that the body becomes solid; bwareaopen removes fur, bedding debris and
% reconstruction noise.
for k = zFirst:zLast
    slice = imfill(body(:, :, k), 'holes');
    body(:, :, k) = bwareaopen(slice, minAreaAxial, 8);
end

% Remove thin spurs -- individual hairs and partial-volume filaments at the
% skin boundary -- that survive the axial pass because they are large within a
% single slice but only a few voxels wide across slices.
%
% NOTE. Each call below passes a 2-D connectivity (8) to an N x 1 x M array.
% MATLAB applies the 3 x 3 neighbourhood to the first two dimensions, and
% because the second one is a singleton the connected components reduce to
% runs along the first axis. That is the intended behaviour here and the
% behaviour the released label was produced with. Rewriting these as true
% in-plane 8-connected openings leaves about 440,000 extra voxels.
for j = 1:ny
    body(:, j, zCleanFirst:zCleanLast) = ...
        bwareaopen(body(:, j, zCleanFirst:zCleanLast), minRun, 8);
end

for i = 1:nx
    body(i, :, zCleanFirst:zCleanLast) = ...
        bwareaopen(body(i, :, zCleanFirst:zCleanLast), minRun, 8);
end

%% --------------------------------------------------------------- 3. Close
% Re-impose the CT support, then fill the holes this reopens.
body = body & (ct.img > 0);
for k = zFirst:zLast
    body(:, :, k) = imfill(body(:, :, k), 'holes');
end

%% ------------------------------------------------------------------- Save
out     = ct;                                   % keep the CT header verbatim
out.img = cast(body, class(ct.img));
save_untouch_nii(out, outFile);

fprintf('%s: %d voxels\n', outFile, nnz(body));
