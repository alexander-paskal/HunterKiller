


def plan_path(occupancy, start_coord, target_coord):
    """
    Plans a path from the start coordinate to the target coordinate using A*
    :param occupancy: an instance of OccupancyGrid
    :param start_coord: 2-dimensional vector, the world frame start coordinate in 2D of the robot
    :param target_coord: 2-dimensional vector, the world frame target coordinate in 2D of the robot
    :return: adjacent coordinate, 2-dimensional coordinate of the adjacent cell
    """
    raise NotImplementedError
