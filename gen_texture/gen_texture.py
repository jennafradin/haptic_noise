"""
Script for generating textures from a CSV file. Sample file is
`gen_texture_example.csv`. Use `add_side_text.blend` to add inscriptions to the
side of the generated textures.
"""

import csv
import gc
import io
import multiprocessing as mp
import os
import pathlib
import tempfile
import time
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import psutil
import pyfqmr
import tqdm.auto
import trimesh as tr
import wx
from matplotlib import pyplot as plt
from PIL import Image
from scipy.stats import norm, truncnorm, uniform
from stl import mesh
from tqdm.contrib import itertools

rng = np.random.default_rng(42)

HEIGHT_MM = 30
WIDTH_MM = 80
DEPTH_MM = 6
CYLINDER_HEIGHT_MM = 0.5

SHOW_PLOT = False

mem = psutil.virtual_memory()
THRESHOLD = 30 * 1024 * 1024 * 1024  # 30GB
if mem.total >= THRESHOLD:
    PPMM = 70
else:
    PPMM = 10


class Distribution(Enum):
    NONE = ""
    NORMAL = "n"
    TRUNCNORM = "tn"
    UNIFORM = "u"
    GAUSSMIX = "gm"


CWD = os.getcwd()
OUTPUT_DIR = os.path.join(CWD, "output")
TEMP_PATH = os.path.join(tempfile.gettempdir(), "temp.stl")


def simplify_mesh(file_name):
    complex_mesh = tr.load_mesh(TEMP_PATH)

    # simplify object
    mesh_simplifier = pyfqmr.Simplify()
    mesh_simplifier.setMesh(complex_mesh.vertices, complex_mesh.faces)
    mesh_simplifier.simplify_mesh(aggressiveness=1, preserve_border=True, verbose=10)

    vertices, faces, normals = mesh_simplifier.getMesh()

    simple_mesh = tr.Trimesh(vertices=vertices, faces=faces)
    simple_mesh.export(file_name)


def generate_mesh(mesh_size, n_triangles, top_array, mesh_scale):
    mesh_shape = mesh.Mesh(np.zeros(n_triangles, dtype=mesh.Mesh.dtype))

    curr_triangle = 0

    # build base
    base_height = int((DEPTH_MM - CYLINDER_HEIGHT_MM) * PPMM)
    vertices = np.array(
        [
            [0, 0, 0],
            [mesh_size[0] - 1, 0, 0],
            [mesh_size[0] - 1, mesh_size[1] - 1, 0],
            [0, mesh_size[1] - 1, 0],
            [0, 0, base_height],
            [mesh_size[0] - 1, 0, base_height],
            [mesh_size[0] - 1, mesh_size[1] - 1, base_height],
            [0, mesh_size[1] - 1, base_height],
        ]
    )
    faces = np.array(
        [
            [0, 3, 1],
            [1, 3, 2],
            [0, 4, 7],
            [0, 7, 3],
            [4, 5, 6],
            [4, 6, 7],
            [5, 1, 2],
            [5, 2, 6],
            [2, 3, 6],
            [3, 7, 6],
            [0, 1, 5],
            [0, 5, 4],
        ]
    )

    for i, f in enumerate(faces):
        for j in range(3):
            mesh_shape.vectors[curr_triangle + i][j] = vertices[f[j], :]
    curr_triangle += 20

    for i, j in itertools.product(
        range(0, mesh_size[0] - 1), range(0, mesh_size[1] - 1)
    ):
        mesh_shape.vectors[curr_triangle][2] = [i, j, top_array[i, j]]
        mesh_shape.vectors[curr_triangle][1] = [i, (j + 1), top_array[i, j + 1]]
        mesh_shape.vectors[curr_triangle][0] = [(i + 1), j, top_array[i + 1, j]]
        mesh_shape.vectors[curr_triangle + 1][0] = [
            (i + 1),
            (j + 1),
            top_array[i + 1, j + 1],
        ]
        mesh_shape.vectors[curr_triangle + 1][1] = [i, (j + 1), top_array[i, j + 1]]
        mesh_shape.vectors[curr_triangle + 1][2] = [(i + 1), j, top_array[i + 1, j]]
        curr_triangle += 2

    mesh_shape.vectors = mesh_shape.vectors * mesh_scale

    mesh_shape.save(TEMP_PATH)


def save_array(arr: np.ndarray, dest: str):
    # print(arr)
    # print(arr.shape)
    with open(dest, "w", encoding="utf-8") as f:
        for row in arr:
            row_str = map(str, row)
            f.write(", ".join(row_str))
            f.write("\n")

    print(f"saved array to file {dest}")


def fig2img(fig: plt.Figure):
    buf = io.BytesIO()

    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)

    return img


def gen_diameters(
    distribution: Distribution, diameters_shape: tuple, params_um: tuple
) -> Tuple[np.ndarray, Optional[Image.Image]]:
    diameters_um: np.ndarray
    sum_rvs: bool = False

    if distribution is Distribution.NONE:
        (diameter_um,) = params_um
        diameters_um = np.ones(diameters_shape) * diameter_um
        return diameters_um, None

    elif distribution is Distribution.NORMAL:
        diameter_um, std_um = params_um
        # diameters_um = rng.normal(diameter_um, std_um, diameters_shape)
        rv = norm(loc=diameter_um, scale=std_um)
        diameters_um = rv.rvs(size=diameters_shape, random_state=rng)

    elif distribution is Distribution.TRUNCNORM:
        diameter_um, std_um, trunc_left, trunc_right = params_um
        a, b = (trunc_left - diameter_um) / std_um, (trunc_right - diameter_um) / std_um
        rv = truncnorm(a, b, loc=diameter_um, scale=std_um)
        diameters_um = rv.rvs(size=diameters_shape, random_state=rng)

    elif distribution is Distribution.GAUSSMIX:
        (
            d1_um,
            std1_um,
            d2_um,
            std2_um,
            trunc_left1,
            trunc_right1,
            trunc_left2,
            trunc_right2,
        ) = params_um
        a1, b1 = (trunc_left1 - d1_um) / std1_um, (trunc_right1 - d1_um) / std1_um
        a2, b2 = (trunc_left2 - d2_um) / std2_um, (trunc_right2 - d2_um) / std2_um

        rv1 = truncnorm(a1, b1, loc=d1_um, scale=std1_um)
        rv2 = truncnorm(a2, b2, loc=d2_um, scale=std2_um)
        sum_rvs = True
        rv = [rv1, rv2]

        d1s_um: np.ndarray = rv1.rvs(size=diameters_shape, random_state=rng)
        d2s_um: np.ndarray = rv2.rvs(size=diameters_shape, random_state=rng)

        # print(f"mean1: {np.mean(d1s_um)}")
        # print(f"std1: {np.std(d1s_um)}")
        # print(f"mean2: {np.mean(d2s_um)}")
        # print(f"std2: {np.std(d2s_um)}")

        diameters_um = np.zeros(diameters_shape)
        p = 0.5
        for i in range(diameters_shape[0]):
            for j in range(diameters_shape[1]):
                if i % 2 == 0:
                    if j % 2 == 0:
                        diameters_um[i, j] = d1s_um[i, j]
                    else:
                        diameters_um[i, j] = d2s_um[i, j]
                else:
                    if j % 2 == 0:
                        diameters_um[i, j] = d2s_um[i, j]
                    else:
                        diameters_um[i, j] = d1s_um[i, j]

    elif distribution is Distribution.UNIFORM:
        start_um, end_um = params_um
        # Using the parameters loc and scale, one obtains the uniform distribution on [loc, loc + scale].
        rv = uniform(loc=start_um, scale=end_um - start_um)
        diameters_um = rv.rvs(size=diameters_shape, random_state=rng)

    # print(f"mean: {np.mean(diameters_um)}")
    # print(f"std: {np.std(diameters_um)}")

    fig, ax = plt.subplots(1, 1)
    ax.hist(
        diameters_um.copy().flatten(),
        density=True,
        bins="auto",
        histtype="stepfilled",
        alpha=0.2,
    )
    x = np.linspace(np.min(diameters_um), np.max(diameters_um), 100)
    # x = np.linspace(rv.ppf(0.001), rv.ppf(0.999), 100)

    if sum_rvs:
        pdf = p * rv[0].pdf(x) + (1 - p) * rv[1].pdf(x)
    else:
        pdf = rv.pdf(x)

    ax.plot(x, pdf, "k-", lw=2, label="frozen pdf")
    if SHOW_PLOT:
        plt.show()

    img = fig2img(fig)

    return diameters_um, img


def parse_distribution_params(distribution: Distribution, params: list[str]):
    ret: tuple

    try:
        if distribution is Distribution.NONE:
            diameter_um = float(params[0])
            ret = (diameter_um,)

        elif distribution is Distribution.NORMAL:
            diameter_um, std_um = float(params[0]), float(params[1])
            ret = (diameter_um, std_um)

        elif distribution is Distribution.TRUNCNORM:
            diameter_um, std_um, trunc_left, trunc_right = (
                float(params[0]),
                float(params[1]),
                float(params[2]),
                float(params[3]),
            )
            ret = (diameter_um, std_um, trunc_left, trunc_right)

        elif distribution is Distribution.GAUSSMIX:
            (
                d1_um,
                std1_um,
                d2_um,
                std2_um,
                trunc_left1,
                trunc_right1,
                trunc_left2,
                trunc_right2,
            ) = (
                float(params[0]),
                float(params[1]),
                float(params[2]),
                float(params[3]),
                float(params[4]),
                float(params[5]),
                float(params[6]),
                float(params[7]),
            )
            ret = (
                d1_um,
                std1_um,
                d2_um,
                std2_um,
                trunc_left1,
                trunc_right1,
                trunc_left2,
                trunc_right2,
            )

        elif distribution is Distribution.UNIFORM:
            start_um, end_um = float(params[0]), float(params[1])
            ret = (start_um, end_um)

        return ret

    except ValueError as err:
        print(
            f"Could not parse parameters {params} for distribution {distribution.value}."
        )
        raise err from None


def create_texture(
    n,
    n_stim_height: int,
    n_stim_width: int,
    distribution: Distribution,
    distribution_parameters: tuple,
):
    params_str = "_".join(map(str, distribution_parameters))
    file_name = f"{n}_{distribution.name.lower()}_{params_str}.stl"
    file_name = os.path.join(OUTPUT_DIR, file_name)
    data_file_name = f"{n}_{distribution.name.lower()}_{params_str}_data.txt"
    data_file_name = os.path.join(OUTPUT_DIR, data_file_name)
    img_file_name = f"{n}_{distribution.name.lower()}_{params_str}_img.png"
    img_file_name = os.path.join(OUTPUT_DIR, img_file_name)

    print(f"output file: {file_name}.")
    print(f"output dir: {OUTPUT_DIR}.")
    time.sleep(1)

    # base layer without texture
    data = int((DEPTH_MM - CYLINDER_HEIGHT_MM) * PPMM) * np.ones(
        (HEIGHT_MM * PPMM, WIDTH_MM * PPMM), dtype=np.uint64
    )

    n_triangles = (data.shape[0] - 1) * (data.shape[1] - 1) * 2 + 20
    # n_triangles = (data.shape[0] - 1) * (data.shape[1] - 1) * 2 * 2 + 20

    # diameters matrix
    diameters_shape = (n_stim_height, n_stim_width)
    diameters_um, fig_img = gen_diameters(
        distribution, diameters_shape, distribution_parameters
    )
    diameters_p = diameters_um / 1000 * PPMM

    save_array(diameters_um, data_file_name)
    if fig_img is not None:
        fig_img.save(img_file_name)

    spacing_x = np.linspace(0, HEIGHT_MM * PPMM, n_stim_height + 2, dtype=np.uint64)[
        1:-1
    ]
    spacing_y = np.linspace(0, WIDTH_MM * PPMM, n_stim_width + 2, dtype=np.uint64)[1:-1]

    for i, j in itertools.product(
        range(diameters_p.shape[0]), range(diameters_p.shape[1])
    ):
        x, y = (spacing_x[i], spacing_y[j])
        x, y = int(x), int(y)
        data[x, y] = int(DEPTH_MM * PPMM)

        angles = np.linspace(0, np.pi / 2, 100)
        for k in angles:
            # for l in np.linspace(0, diameters[i, j], max_diam):
            #     xc, yc = (x + np.sin(k) * l, y + np.cos(k) * l)
            #     data[int(xc), int(yc)] = 255
            xc, yc = (np.sin(k) * diameters_p[i, j], np.cos(k) * diameters_p[i, j])
            xc, yc = int(xc), int(yc)
            data[x : x + xc, y : y + yc] = int(DEPTH_MM * PPMM)
            data[x - xc : x, y : y + yc] = int(DEPTH_MM * PPMM)
            data[x - xc : x, y - yc : y] = int(DEPTH_MM * PPMM)
            data[x : x + xc, y - yc : y] = int(DEPTH_MM * PPMM)

    top_array = data
    top_array = np.rot90(top_array, -1, (0, 1))

    mesh_size = top_array.shape

    mesh_scale = 1 / PPMM

    p1 = mp.Process(
        target=generate_mesh,
        args=(mesh_size, n_triangles, top_array, mesh_scale),
    )
    p1.start()
    p1.join()

    gc.collect()

    p2 = mp.Process(target=simplify_mesh, args=(file_name,))
    p2.start()
    p2.join()


def main() -> int:
    base_n = 0

    print("Select experiment file")
    _ = wx.App()
    frm = wx.Frame(None, title="Haptic Noise")

    filename: str
    # otherwise ask the user what new file to open
    with wx.FileDialog(
        frm,
        "Open CSV file",
        wildcard="CSV files (*.csv)|*.csv|All files (*.*)|*.*",
        style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
    ) as fileDialog:
        if fileDialog.ShowModal() == wx.ID_CANCEL:
            return 1  # the user changed their mind

        filename = fileDialog.GetPath()

    print("Selected file: ", filename)

    pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    errors = []

    with open(filename, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for i, row in tqdm.tqdm(enumerate(reader)):
            if i == 0:  # field names
                continue
            n_x = int(row[0])
            n_y = int(row[1])

            try:
                distribution = Distribution[row[2].upper()]
            except KeyError:
                print(f"Invalid distribution value: '{row[2]}'")
                errors.append((i, row))
                continue

            params = row[3:]
            distribution_params = parse_distribution_params(distribution, params)
            create_texture(base_n + i, n_y, n_x, distribution, distribution_params)

    if len(errors) > 0:
        print("Errors:")
        for i, row in errors:
            print(f"{i}: {row}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
