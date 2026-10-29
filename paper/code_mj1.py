    def mj1(self):
        # the palm is driven kinematically: its position and velocity are imposed every step
        self.data.qpos[self._pq] = self.palm_z - 0.5; self.data.qvel[self._pv] = self.palm_v
        if self.ratio != 1.0 and self.armed:
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self._rod, self._vel, 0)
            v = abs(self._vel[5] - self.palm_v)
            if not self.sliding and v > self.v_slip:
                self.sliding = True; self.n_breakaway += 1
                self.model.geom_friction[self._g, 0] = self.mu_k
            elif self.sliding and v < self.v_stick:
                self.sliding = False
                self.model.geom_friction[self._g, 0] = self.mu_s
        mujoco.mj_step(self.model, self.data)
        self.data.qpos[self._pq] = self.palm_z - 0.5; self.data.qvel[self._pv] = self.palm_v
