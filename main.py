



class PathFollowingRobot:
    def run(self):

        while True:
            self.update_location()
            waypoint = self.get_waypoint()

            if self.reached(waypoint):
                self.update_waypoint()
                waypoint = self.get_waypoint

            next_coord = self.plan_path(waypoint)
            control = self.compute_control(next_coord)
            self.apply_control(control)
            self.wait()



class SlamRobot:
    def run(self):

        while True:
            self.update_location()
            waypoint = self.get_waypoint()

            if self.reached(waypoint):
                self.update_waypoint()
                waypoint = self.get_waypoint

            next_coord = self.plan_path(waypoint)
            control = self.compute_control(next_coord)
            self.apply_control(control)

            self.predict_pose(control)

            lidar = self.get_lidar()
            self.update_map(lidar)
            self.update_pose(lidar)


