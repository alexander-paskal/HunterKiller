"""
Defines an OccupancyGrid class, which represents a 2D probabilistic occupancy grid
where the values correspond to the log odds probability of a given section of the map being
occupied

"""
import numpy as np


class OccupancyGrid:
    def __init__(self, dims, resolution):
        raise NotImplementedError

    def serialize(self):
        raise NotImplementedError

    def coord2grid(self, coords):
        """
        Converts an array of 2D coordinates in the world frame
        to an array of unique grid indices for the corresponding cell in the
        occupancy grid
        :param coords: Nx2 array of 2-dimensional real-valued coordinates
        :return: inds: Mx2 array of unique [i, j] indices, where M <= N
        """
        raise NotImplementedError

    def grid2coord(self, inds):
        """
        Converts an array of grid indices to 2d coordinates
        :param inds:
        :return: coords: Nx2 array of 2 dimensional coordinates
        """
        raise NotImplementedError

    def binary(self):
        """
        Returns a binary occupancy grid
        :return:
        """
        raise NotImplementedError

