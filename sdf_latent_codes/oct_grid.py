import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
import open3d as o3d
from torch.autograd import Variable

# Store grads for normals
grads = {}
# Hook definition
def save_grad(name):
    def hook(grad):
        grads[name] = grad

    return hook
def get_grid_surface_hook(feat_sdf, lods, octgrid, sdfnet):
    with torch.no_grad():
        xyz_0 = np.array(octgrid.centers)[np.array(octgrid.level) == lods[0]]
        xyz = torch.from_numpy(xyz_0).to(feat_sdf.device)
        for lod in lods:
            inputs_sdfnet = torch.cat([feat_sdf.expand(xyz.shape[0], -1), xyz], 1).to(feat_sdf.device, feat_sdf.dtype)
            sdf = sdfnet(inputs_sdfnet)
            occ = sdf.abs() < get_cell_size(lod)
            xyz = subdivide(xyz[occ[:, 0]], level=lod)
    points = Variable(xyz.to(feat_sdf.device, feat_sdf.dtype), requires_grad=True)
    points.register_hook(save_grad('grid_points'))
    return points

class OctGrid:
    def __init__(self, subdiv=2):
        # branch: 0 1 2 3 4 5 6 7
        # x:      - - - - + + + +
        # y:      - - + + - - + +
        # z:      - + - + - + - +

        self.centers = [(0, 0, 0)]
        self.level = [0]

        for s in range(subdiv):
            self.subdivide()

        # get points and lods
        self.points, self.points_level = self.get_points()

        # print('Level: {}, Points: {}'.format(self.points_level.max(), len(self.points)))

    def subdivide(self, max_level=7):

        # Add level points
        centers_added, level_added = [], []

        # Get current latest level
        level_max = max(self.level)
        centers = np.array(self.centers)[np.array(self.level) == level_max]

        for id, pos in enumerate(centers):

            side = (1 / (2 ** level_max)) * 2
            offset = side / 4
            # radius = ((np.sqrt(2) * side) / 2)

            # Add new points
            point_0 = (pos[0] - offset, pos[1] - offset, pos[2] - offset)
            point_1 = (pos[0] - offset, pos[1] - offset, pos[2] + offset)
            point_2 = (pos[0] - offset, pos[1] + offset, pos[2] - offset)
            point_3 = (pos[0] - offset, pos[1] + offset, pos[2] + offset)
            point_4 = (pos[0] + offset, pos[1] - offset, pos[2] - offset)
            point_5 = (pos[0] + offset, pos[1] - offset, pos[2] + offset)
            point_6 = (pos[0] + offset, pos[1] + offset, pos[2] - offset)
            point_7 = (pos[0] + offset, pos[1] + offset, pos[2] + offset)
            centers_added.extend([point_0, point_1, point_2, point_3, point_4, point_5, point_6, point_7])
            level_added.extend([level_max + 1] * 8)

        # Store new points to global list
        if not self.level[-1] == max_level:
            self.centers.extend(centers_added)
            self.level.extend(level_added)


    def build_boxes(self):
        linesets = []
        points, lines, colors = [], [], []
        for id, (pos, lev) in enumerate(zip(self.centers, self.level)):
            offset = 1 / (2 ** lev)

            # Get point coordinates
            point_0 = (pos[0] - offset, pos[1] - offset, pos[2] - offset)
            point_1 = (pos[0] - offset, pos[1] - offset, pos[2] + offset)
            point_2 = (pos[0] - offset, pos[1] + offset, pos[2] - offset)
            point_3 = (pos[0] - offset, pos[1] + offset, pos[2] + offset)
            point_4 = (pos[0] + offset, pos[1] - offset, pos[2] - offset)
            point_5 = (pos[0] + offset, pos[1] - offset, pos[2] + offset)
            point_6 = (pos[0] + offset, pos[1] + offset, pos[2] - offset)
            point_7 = (pos[0] + offset, pos[1] + offset, pos[2] + offset)

            points.extend([point_0, point_1, point_2, point_3, point_4, point_5, point_6, point_7])
            line_inc = ((len(points) // 8) - 1) * 8
            lines.extend((np.array([[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7], [0, 4], [1, 5], [2, 6], [3, 7]]) + line_inc).tolist())
            # colors.extend([[prob, prob, prob]] * 8)

        lineset = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        # lineset.colors = o3d.utility.Vector3dVector([[prob, prob, prob]] * 8)

        # linesets.append(lineset)

        return lineset

    def get_points(self):
        points = []
        points_level = []
        for id, (pos, lev) in enumerate(zip(self.centers, self.level)):
            offset = 1 / (2 ** lev)

            # Get point coordinates
            point_0 = (pos[0] - offset, pos[1] - offset, pos[2] - offset)
            point_1 = (pos[0] - offset, pos[1] - offset, pos[2] + offset)
            point_2 = (pos[0] - offset, pos[1] + offset, pos[2] - offset)
            point_3 = (pos[0] - offset, pos[1] + offset, pos[2] + offset)
            point_4 = (pos[0] + offset, pos[1] - offset, pos[2] - offset)
            point_5 = (pos[0] + offset, pos[1] - offset, pos[2] + offset)
            point_6 = (pos[0] + offset, pos[1] + offset, pos[2] - offset)
            point_7 = (pos[0] + offset, pos[1] + offset, pos[2] + offset)

            points.extend([point_0, point_1, point_2, point_3, point_4, point_5, point_6, point_7])
            points_level.extend([lev, lev, lev, lev, lev, lev, lev, lev])

        # points_unique, ids_unique = np.unique(np.array(points), axis=0, return_index=True)
        # level_points_unique = level_points[ids_unique]

        # get lods
        points_unique = np.unique(np.array(points), axis=0)
        octgrids_dict = defaultdict(defaultdict(dict).copy)
        for p in points_unique:
            octgrids_dict[p[0]][p[1]][p[2]] = 100
        for p, l in zip(points, points_level):
            octgrids_dict[p[0]][p[1]][p[2]] = min(octgrids_dict[p[0]][p[1]][p[2]], l)

        lods = np.zeros_like(points_unique[:, 0])
        for id, p in enumerate(points_unique):
            lods[id] = octgrids_dict[p[0]][p[1]][p[2]]

        return points_unique, lods

    def get_lods(self):
        octgrids_dict = defaultdict(dict)
        for p in self.points:
            octgrids_dict[p[0]][p[1]][p[2]] = 100
        for p, l in zip(self.points, self.points_level):
            octgrids_dict[p[0]][p[1]][p[2]] = min(octgrids_dict[p[0]][p[1]][p[2]], l)

        lods = np.zeros_like

        return octgrids_dict

    def find_parent(self, points, level=1):

        cells = []
        boxes_o3d = []

        for pos in points:

            center = (0, 0, 0)

            for lev in range(level):

                side = (1 / (2 ** lev)) * 2
                offset = side / 4

                ## find out which direction we're heading in
                branch = self.__findBranch(center, pos)

                if branch == 0:
                    center = (center[0] - offset, center[1] - offset, center[2] - offset)
                elif branch == 1:
                    center = (center[0] - offset, center[1] - offset, center[2] + offset)
                elif branch == 2:
                    center = (center[0] - offset, center[1] + offset, center[2] - offset)
                elif branch == 3:
                    center = (center[0] - offset, center[1] + offset, center[2] + offset)
                elif branch == 4:
                    center = (center[0] + offset, center[1] - offset, center[2] - offset)
                elif branch == 5:
                    center = (center[0] + offset, center[1] - offset, center[2] + offset)
                elif branch == 6:
                    center = (center[0] + offset, center[1] + offset, center[2] - offset)
                elif branch == 7:
                    center = (center[0] + offset, center[1] + offset, center[2] + offset)

            points_cell = self.__getCell(center, level)
            cells.append(points_cell)

        cells = np.unique(np.array(cells), axis=0)
        for c in cells:
            bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(c))
            bbox.color = [0, 0, 0]
            boxes_o3d.append(bbox)

        # o3d.visualization.draw_geometries(boxes_o3d)

        return boxes_o3d

    def get_annotations(self, points, level=1):

        cells = []
        boxes_o3d = []

        for pos in points:

            center = (0, 0, 0)

            for lev in range(level):

                side = (1 / (2 ** lev)) * 2
                offset = side / 4

                ## find out which direction we're heading in
                branch = self.__findBranch(center, pos)

                if branch == 0:
                    center = (center[0] - offset, center[1] - offset, center[2] - offset)
                elif branch == 1:
                    center = (center[0] - offset, center[1] - offset, center[2] + offset)
                elif branch == 2:
                    center = (center[0] - offset, center[1] + offset, center[2] - offset)
                elif branch == 3:
                    center = (center[0] - offset, center[1] + offset, center[2] + offset)
                elif branch == 4:
                    center = (center[0] + offset, center[1] - offset, center[2] - offset)
                elif branch == 5:
                    center = (center[0] + offset, center[1] - offset, center[2] + offset)
                elif branch == 6:
                    center = (center[0] + offset, center[1] + offset, center[2] - offset)
                elif branch == 7:
                    center = (center[0] + offset, center[1] + offset, center[2] + offset)

            points_cell = self.__getCell(center, level)
            cells.append(points_cell)

        cells = np.unique(np.array(cells), axis=0)
        for c in cells:
            bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(c))
            bbox.color = [0, 0, 0]
            boxes_o3d.append(bbox)

        # o3d.visualization.draw_geometries(boxes_o3d)

        return boxes_o3d


    @staticmethod
    def __getCell(pos, level):

        side = (1 / (2 ** level)) * 2
        offset = side / 2

        points = []
        # Get point coordinates
        point_0 = (pos[0] - offset, pos[1] - offset, pos[2] - offset)
        point_1 = (pos[0] - offset, pos[1] - offset, pos[2] + offset)
        point_2 = (pos[0] - offset, pos[1] + offset, pos[2] - offset)
        point_3 = (pos[0] - offset, pos[1] + offset, pos[2] + offset)
        point_4 = (pos[0] + offset, pos[1] - offset, pos[2] - offset)
        point_5 = (pos[0] + offset, pos[1] - offset, pos[2] + offset)
        point_6 = (pos[0] + offset, pos[1] + offset, pos[2] - offset)
        point_7 = (pos[0] + offset, pos[1] + offset, pos[2] + offset)

        points.extend([point_0, point_1, point_2, point_3, point_4, point_5, point_6, point_7])

        return points

    @staticmethod
    def __findBranch(center, position):
        """
        helper function
        returns an index corresponding to a branch
        pointing in the direction we want to go
        """
        index = 0
        if (position[0] >= center[0]):
            index |= 4
        if (position[1] >= center[1]):
            index |= 2
        if (position[2] >= center[2]):
            index |= 1
        return index

    @staticmethod
    def subdivide_given(points, level=6):

        # Add level points
        centers_added = []

        for id, pos in enumerate(points.tolist()):

            side = (1 / (2 ** level)) * 2
            offset = side / 4

            # Add new points
            point_0 = (pos[0] - offset, pos[1] - offset, pos[2] - offset)
            point_1 = (pos[0] - offset, pos[1] - offset, pos[2] + offset)
            point_2 = (pos[0] - offset, pos[1] + offset, pos[2] - offset)
            point_3 = (pos[0] - offset, pos[1] + offset, pos[2] + offset)
            point_4 = (pos[0] + offset, pos[1] - offset, pos[2] - offset)
            point_5 = (pos[0] + offset, pos[1] - offset, pos[2] + offset)
            point_6 = (pos[0] + offset, pos[1] + offset, pos[2] - offset)
            point_7 = (pos[0] + offset, pos[1] + offset, pos[2] + offset)
            centers_added.extend([point_0, point_1, point_2, point_3, point_4, point_5, point_6, point_7])

        return centers_added

    def get_surface_points_given_hook(self, sdf, pcd, threshold=0.03, graph=True):
        """
        Zero isosurface projection

        Args:
            pred_sdf_grid (N,1): output of DeepSDF
            threshold (float): band of points to be projected onto the surface

        Returns: projected points (N,3), NOCS (N,3), normals (N,3)
        """
        sdf.sum().backward(retain_graph=True)
        normals = F.normalize(grads['grid_points'][:, :], dim=-1).detach()
        # Project points onto the surface
        points = pcd - (sdf * normals)
        # Keep only SDF points close to the surface
        surface_mask = sdf.abs() < threshold
        points_masked = points.masked_select(surface_mask).view(-1, 3)
        normals_masked = normals.masked_select(surface_mask).view(-1, 3)
        points_masked_normed = (points_masked + 1) / 2
        return points_masked.to(sdf.to(sdf.dtype)), normals_masked.to(sdf.to(sdf.dtype))

    def get_surface_points_given(self, sdf, pcd, threshold=0.03, graph=True):
        """
        Zero isosurface projection

        Args:
            pred_sdf_grid (N,1): output of DeepSDF
            threshold (float): band of points to be projected onto the surface

        Returns: projected points (N,3), NOCS (N,3), normals (N,3)

        """
        normals_single, = torch.autograd.grad(sdf.sum(), pcd, retain_graph=True)
        normals = F.normalize(normals_single, dim=-1).detach()
        # normals_single_normed[normals_single_normed != normals_single_normed] = 0

        # Project points onto the surface
        points = pcd - (sdf * normals)
        # Keep only SDF points close to the surface
        surface_mask = sdf.abs() < threshold
        points_masked = points.masked_select(surface_mask).view(-1, 3)
        normals_masked = normals.masked_select(surface_mask).view(-1, 3)
        points_masked_normed = (points_masked + 1) / 2
        return points_masked.to(sdf.to(sdf.dtype)), normals_masked.to(sdf.to(sdf.dtype))

    def get_surface_points_sparse(self, sdf, pcd, threshold=0.02, graph=True):
        """
        Zero isosurface projection

        Args:
            pred_sdf_grid (N,1): output of DeepSDF
            threshold (float): band of points to be projected onto the surface

        Returns: projected points (N,3), NOCS (N,3), normals (N,3)

        """
        # Keep only SDF points close to the surface
        surface_mask = sdf.abs() < threshold
        points_masked = pcd.masked_select(surface_mask).view(-1, 3)
        # normals_masked = normals.masked_select(surface_mask).view(-1, 3)
        # nocs = (points_masked + 1) / 2
        return points_masked.to(sdf.to(sdf.dtype))

def get_cell_size(lod=6):
    return (1 / (2 ** lod)) * 2


def build_boxes(centers, level):
    points, lines, colors = [], [], []
    for id, pos in enumerate(centers):
        offset = 1 / (2 ** level)

        # Get point coordinates
        point_0 = (pos[0] - offset, pos[1] - offset, pos[2] - offset)
        point_1 = (pos[0] - offset, pos[1] - offset, pos[2] + offset)
        point_2 = (pos[0] - offset, pos[1] + offset, pos[2] - offset)
        point_3 = (pos[0] - offset, pos[1] + offset, pos[2] + offset)
        point_4 = (pos[0] + offset, pos[1] - offset, pos[2] - offset)
        point_5 = (pos[0] + offset, pos[1] - offset, pos[2] + offset)
        point_6 = (pos[0] + offset, pos[1] + offset, pos[2] - offset)
        point_7 = (pos[0] + offset, pos[1] + offset, pos[2] + offset)

        points.extend([point_0, point_1, point_2, point_3, point_4, point_5, point_6, point_7])
        line_inc = ((len(points) // 8) - 1) * 8
        lines.extend((np.array([[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7], [0, 4], [1, 5], [2, 6], [3, 7]]) + line_inc).tolist())
        # colors.extend([[prob, prob, prob]] * 8)

    lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    return lineset


def subdivide(centers, level):
    offset_size = (1 / (2 ** level)) * 2 / 4
    offsets = torch.tensor([(-1, -1, -1), (-1, -1, 1), (-1, +1, -1), (-1, 1, 1), (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)]).to(centers.device) * offset_size
    centers_new = centers.repeat_interleave(8, dim=-2) + offsets.repeat(centers.shape[0], 1)

    return centers_new


def get_corners(centers, level):
    points = []
    for id, pos in enumerate(centers, level):
        offset = 1 / (2 ** level)

        # Get point coordinates
        point_0 = (pos[0] - offset, pos[1] - offset, pos[2] - offset)
        point_1 = (pos[0] - offset, pos[1] - offset, pos[2] + offset)
        point_2 = (pos[0] - offset, pos[1] + offset, pos[2] - offset)
        point_3 = (pos[0] - offset, pos[1] + offset, pos[2] + offset)
        point_4 = (pos[0] + offset, pos[1] - offset, pos[2] - offset)
        point_5 = (pos[0] + offset, pos[1] - offset, pos[2] + offset)
        point_6 = (pos[0] + offset, pos[1] + offset, pos[2] - offset)
        point_7 = (pos[0] + offset, pos[1] + offset, pos[2] + offset)

        points.extend([point_0, point_1, point_2, point_3, point_4, point_5, point_6, point_7])

    # get points
    points_unique = np.unique(np.array(points), axis=0)

    return points_unique


def get_grid_surface(feat_sdf, lods, octgrid, sdfnet):
    with torch.no_grad():
        xyz_0 = np.array(octgrid.centers)[np.array(octgrid.level) == lods[0]]
        xyz = torch.from_numpy(xyz_0).to(feat_sdf.device)
        for lod in lods:
            inputs_sdfnet = torch.cat([feat_sdf.expand(xyz.shape[0], -1), xyz], 1).to(feat_sdf.device, feat_sdf.dtype)
            sdf = sdfnet(inputs_sdfnet)
            occ = sdf.abs() < get_cell_size(lod)
            xyz = subdivide(xyz[occ[:, 0]], level=lod)
    points = Variable(xyz.to(feat_sdf.device, feat_sdf.dtype), requires_grad=True)
    return points
