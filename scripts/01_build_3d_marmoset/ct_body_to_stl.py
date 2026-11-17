#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Micro-CT (NIfTI) -> body-surface label -> smoothed label -> surface mesh (STL).

This script reproduces, in a single command, the whole segmentation pipeline
that produced the 3D marmoset body model used in the paper.  The original
processing was done with MATLAB scripts (steps 1-2, see matlab_original/)
followed by manual operations in the 3D Slicer Segment Editor (steps 3-4).
This script removes both the MATLAB and the GUI dependency.

Steps
  1. Crop        Mask the CT with a cylindrical field-of-view template and with
                 a label of the cylindrical holder tube, and restrict the axial
                 range, so that only the animal remains.
  2. Clean       Per-slice hole filling and small-component removal in the axial
                 planes, then removal of short runs along x and y.
  3. Smooth      Gaussian smoothing of the binary label followed by
                 re-thresholding at 0.5 (equivalent to the 3D Slicer Segment
                 Editor "Smoothing / Gaussian" effect).
  4. Mesh        Marching cubes on the smoothed scalar field at the 0.5
                 iso-level, written as a binary STL in LPS world coordinates
                 (the convention used by 3D Slicer), so that the mesh overlays
                 the input CT without any further transform.

Inputs (all on the same 512 x 512 x 1334 grid, 0.196 x 0.196 x 0.3 mm)
  --ct        whole-body micro-CT, Hounsfield units, int16
  --fov       field-of-view template; one axial slice of it (--fov-slice)
              defines the circular reconstruction area to keep
  --holder    label of the cylindrical holder tube; one axial slice of it
              (--holder-slice) is extruded along z and removed

Outputs
  <out>/body_mask.nii.gz            binary body label, same affine as the CT
  <out>/body_mask_smoothed.nii.gz   after Gaussian smoothing + 0.5 threshold
  <out>/body_surface.stl            binary STL, LPS, mm

Usage
  python ct_body_to_stl.py \
      --ct     20221020_173232BFbBs003a003.nii \
      --fov    body_sample.nii.gz \
      --holder tst2Segmentation-Segment_1-label.nii.gz \
      --out    output/

Requirements: numpy, scipy, nibabel, vtk  (see ../../environment.yml)
"""
from __future__ import print_function

import argparse
import os
import time

import nibabel as nib
import numpy as np
import vtk
from scipy import ndimage as ndi
from vtk.util import numpy_support as ns

# 8-connectivity in a plane; matches MATLAB bwareaopen(BW, P, 8) on a 2-D array.
CONN2D = np.ones((3, 3), bool)

# Connectivity along the first axis only.  The original MATLAB code calls
# bwareaopen(y(:, j, z0:z1), 10, 8), i.e. it passes a 2-D connectivity to an
# N x 1 x M array.  MATLAB then applies the 3x3 neighbourhood to the first two
# dimensions, and because the second one is a singleton the components reduce
# to runs along the first axis.  Reproducing that exactly matters: using true
# in-plane 8-connectivity here leaves ~440,000 extra voxels (Dice 0.990
# against the released label instead of 1.000).
CONN1D = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], bool)

_T0 = time.time()


def log(*a):
    print('[%6.1fs]' % (time.time() - _T0), *a)


def area_open(bw, min_size, structure=CONN2D):
    """Remove connected components smaller than min_size.

    Equivalent to MATLAB bwareaopen(BW, min_size, conn).
    """
    lab, n = ndi.label(bw, structure=structure)
    if n == 0:
        return bw
    cnt = np.bincount(lab.ravel())
    cnt[0] = 0
    return (cnt >= min_size)[lab]


def fill_holes(bw):
    """Equivalent to MATLAB imfill(BW, 'holes')."""
    return ndi.binary_fill_holes(bw)


# --------------------------------------------------------------------------
# 1. Crop: field of view, holder tube, axial range
# --------------------------------------------------------------------------
def crop(ct, fov, holder, fov_slice, holder_slice, z_first, z_last):
    inside_fov = fov[:, :, fov_slice] > 0          # circular reconstruction area
    is_holder = holder[:, :, holder_slice] > 0     # holder tube cross-section

    keep2d = inside_fov & (~is_holder)             # extruded along z
    body = (ct > 0) & keep2d[:, :, None]

    body[:, :, :z_first] = False                   # above the head / below the tail
    body[:, :, z_last + 1:] = False
    log('step 1  crop            %10d voxels' % body.sum())
    return body


# --------------------------------------------------------------------------
# 2. Clean: fill holes, drop small axial components, drop short runs along x and y
# --------------------------------------------------------------------------
def clean(body, ct, z_first_clean, z_last_clean, min_axial, min_run):
    nx, ny, nz = body.shape

    for k in range(nz):                            # axial planes
        if not body[:, :, k].any():
            continue
        s = fill_holes(body[:, :, k])
        body[:, :, k] = area_open(s, min_axial)
    log('step 2a axial clean     %10d voxels' % body.sum())

    zs = slice(z_first_clean, z_last_clean + 1)
    for j in range(ny):                            # runs along x, see CONN1D
        s = body[:, j, zs]
        if s.any():
            body[:, j, zs] = area_open(s, min_run, CONN1D)
    log('step 2b x-run clean     %10d voxels' % body.sum())

    for i in range(nx):                            # runs along y, see CONN1D
        s = body[i, :, zs]
        if s.any():
            body[i, :, zs] = area_open(s, min_run, CONN1D)
    log('step 2c y-run clean     %10d voxels' % body.sum())

    body &= ct > 0                                 # re-impose the CT support
    for k in range(nz):                            # final axial hole filling
        if body[:, :, k].any():
            body[:, :, k] = fill_holes(body[:, :, k])
    log('step 2d final fill      %10d voxels' % body.sum())
    return body


# --------------------------------------------------------------------------
# 3. Smooth: Gaussian on the binary label, re-threshold at 0.5
# --------------------------------------------------------------------------
def smooth(body, voxel_mm, sigma_voxels, sigma_mm, pad):
    """Return (smoothed_binary_full_grid, scalar_field_crop, crop_origin_index).

    `sigma_voxels` is an isotropic standard deviation in voxel units, which is
    what reproduces the released model; passing `sigma_mm` instead makes the
    standard deviation isotropic in millimetres, which is the right choice for
    a volume with a different voxel aspect ratio.

    The Gaussian is evaluated on a bounding box of the label (plus `pad`
    voxels) to keep memory and run time bounded; the field is zero outside.
    """
    sl = ndi.find_objects(body.astype(np.uint8))[0]
    sl = tuple(slice(max(0, s.start - pad), min(n, s.stop + pad))
               for s, n in zip(sl, body.shape))
    voxel_mm = np.asarray(voxel_mm, float)
    if sigma_mm is not None:
        sigma_vox = sigma_mm / voxel_mm
    else:
        sigma_vox = np.full(3, float(sigma_voxels))
    log('step 3  gaussian sigma = %s voxels = %s mm, bbox %s'
        % (np.round(sigma_vox, 4).tolist(),
           np.round(sigma_vox * voxel_mm, 4).tolist(),
           [(s.start, s.stop) for s in sl]))

    field = ndi.gaussian_filter(body[sl].astype(np.float64), sigma=sigma_vox)
    out = np.zeros(body.shape, bool)
    out[sl] = field > 0.5
    log('step 3  smoothed        %10d voxels' % out.sum())
    return out, field, np.array([s.start for s in sl])


# --------------------------------------------------------------------------
# 4. Mesh: marching cubes at 0.5, binary STL in LPS
# --------------------------------------------------------------------------
def write_stl(field, offset_index, affine, path, level=0.5,
              sinc_iterations=0, sinc_passband=0.1):
    """Write the `level` iso-surface of `field` as a binary STL in LPS mm.

    `affine` maps the *full* voxel grid to RAS (NIfTI convention); `field` is a
    crop of that grid starting at voxel `offset_index`.  STL files written by
    3D Slicer are in LPS, so x and y are negated.  The j axis is reversed as
    well, which turns index -> LPS into a positive diagonal scaling and
    therefore keeps the triangle winding (and the surface normals) correct.
    """
    lps = np.diag([-1., -1., 1., 1.]).dot(affine)
    nx, ny, nz = field.shape
    field = field[:, ::-1, :]

    spacing = (lps[0, 0], -lps[1, 1], lps[2, 2])
    o = lps[:3, :3].dot(offset_index) + lps[:3, 3]   # crop origin in LPS
    origin = (o[0], lps[1, 1] * (ny - 1) + o[1], o[2])
    log('step 4  LPS origin %s spacing %s'
        % (np.round(origin, 4).tolist(), np.round(spacing, 4).tolist()))

    img = vtk.vtkImageData()
    img.SetDimensions(nx, ny, nz)
    img.SetSpacing(*spacing)
    img.SetOrigin(*origin)
    flat = np.ascontiguousarray(field.transpose(2, 1, 0)).ravel()  # x fastest
    img.GetPointData().SetScalars(
        ns.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT))
    del flat

    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(img)
    mc.SetValue(0, level)
    mc.ComputeNormalsOff()
    mc.ComputeGradientsOff()
    mc.ComputeScalarsOff()
    mc.Update()
    pd = mc.GetOutput()
    log('step 4  marching cubes  %10d points %10d triangles'
        % (pd.GetNumberOfPoints(), pd.GetNumberOfCells()))

    if sinc_iterations > 0:
        sm = vtk.vtkWindowedSincPolyDataFilter()
        sm.SetInputData(pd)
        sm.SetNumberOfIterations(sinc_iterations)
        sm.SetPassBand(sinc_passband)
        sm.BoundarySmoothingOff()
        sm.FeatureEdgeSmoothingOff()
        sm.NonManifoldSmoothingOn()
        sm.NormalizeCoordinatesOn()
        sm.Update()
        pd = sm.GetOutput()

    nrm = vtk.vtkPolyDataNormals()
    nrm.SetInputData(pd)
    nrm.ConsistencyOn()
    nrm.SplittingOff()
    nrm.AutoOrientNormalsOn()
    nrm.Update()

    w = vtk.vtkSTLWriter()
    w.SetInputData(nrm.GetOutput())
    w.SetFileName(path)
    w.SetFileTypeToBinary()
    w.SetHeader('3D Slicer output. SPACE=LPS')
    w.Write()
    log('step 4  wrote %s' % path)


def save_nifti(mask, affine, path):
    img = nib.Nifti1Image(mask.astype(np.uint8), affine)
    img.header.set_qform(affine, code=1)
    img.header.set_sform(affine, code=1)
    img.header.set_xyzt_units('mm')
    img.header.set_data_dtype(np.uint8)
    nib.save(img, path)
    log('wrote %s' % path)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ct', required=True, help='whole-body micro-CT (NIfTI, HU)')
    p.add_argument('--fov', required=True, help='field-of-view template (NIfTI)')
    p.add_argument('--holder', required=True, help='holder-tube label (NIfTI)')
    p.add_argument('--out', default='.', help='output directory')

    p.add_argument('--fov-slice', type=int, default=674,
                   help='0-based axial index of the FOV template slice to extrude')
    p.add_argument('--holder-slice', type=int, default=114,
                   help='0-based axial index of the holder slice to extrude')
    p.add_argument('--z-range', type=int, nargs=2, default=(83, 1254),
                   metavar=('FIRST', 'LAST'),
                   help='0-based inclusive axial range of the animal')
    p.add_argument('--clean-z-range', type=int, nargs=2, default=(199, 1099),
                   metavar=('FIRST', 'LAST'),
                   help='0-based inclusive axial range for the coronal/sagittal cleaning')
    p.add_argument('--min-axial', type=int, default=100,
                   help='minimum component size in an axial plane (voxels)')
    p.add_argument('--min-run', type=int, default=10,
                   help='minimum run length along x and y (voxels)')
    p.add_argument('--sigma-voxels', type=float, default=1.04,
                   help='Gaussian smoothing sigma, isotropic in voxel units '
                        '(default 1.04, which reproduces the released model)')
    p.add_argument('--sigma-mm', type=float, default=None,
                   help='Gaussian smoothing sigma in mm; overrides --sigma-voxels')
    p.add_argument('--pad', type=int, default=8,
                   help='bounding-box padding in voxels for the Gaussian/meshing')
    p.add_argument('--sinc-iterations', type=int, default=0,
                   help='optional windowed-sinc mesh smoothing (0 = off)')
    a = p.parse_args()

    if not os.path.isdir(a.out):
        os.makedirs(a.out)

    nii = nib.load(a.ct)
    affine = nii.affine
    voxel_mm = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    ct = np.asanyarray(nii.dataobj)
    fov = np.asanyarray(nib.load(a.fov).dataobj)
    holder = np.asanyarray(nib.load(a.holder).dataobj)
    log('CT %s %s, voxel %s mm' % (ct.shape, ct.dtype, np.round(voxel_mm, 4).tolist()))
    for name, arr in (('fov', fov), ('holder', holder)):
        if arr.shape != ct.shape:
            raise SystemExit('%s grid %s does not match the CT grid %s'
                             % (name, arr.shape, ct.shape))

    body = crop(ct, fov, holder, a.fov_slice, a.holder_slice, *a.z_range)
    body = clean(body, ct, a.clean_z_range[0], a.clean_z_range[1],
                 a.min_axial, a.min_run)
    save_nifti(body, affine, os.path.join(a.out, 'body_mask.nii.gz'))

    smoothed, field, offset = smooth(body, voxel_mm, a.sigma_voxels,
                                     a.sigma_mm, a.pad)
    save_nifti(smoothed, affine, os.path.join(a.out, 'body_mask_smoothed.nii.gz'))

    write_stl(field, offset, affine, os.path.join(a.out, 'body_surface.stl'),
              level=0.5, sinc_iterations=a.sinc_iterations)
    log('done')


if __name__ == '__main__':
    main()
