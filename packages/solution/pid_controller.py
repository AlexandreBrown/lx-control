#!/usr/bin/env python3

from typing import Tuple
import numpy as np

class PIDController():
    def __init__(self):

        # We will initialize some variables that might be useful
        self.prev_e_heading = 0.0
        self.prev_e_offset = 0.0
        self.prev_int_heading = 0.0
        self.prev_int_offset = 0.0

        self.kp = 0.0
        self.ki = 0.0
        self.kd = 0.0
    
    def _wrap_angle(self, angle: float) -> float:
        """wrap an angle to [-pi, pi]"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def HeadingControl(self,
                       v_ref: float,
                       theta_ref: float,
                       theta_curr: float,
                       delta_t: float
    ) -> Tuple[float, float]:
        """
        PID performing heading control.
        Args:
            v_ref:      reference velocity.
            theta_ref:  reference heading pose.
            theta_curr: the current estimated heading.
            delta_t:    time interval since last call.
        Returns:
            v:          linear velocity of the Duckiebot
            omega:      angular velocity of the Duckiebot
        """
        e_t = self._wrap_angle(theta_ref - theta_curr)

        self.prev_int_heading += np.clip(e_t * delta_t, -0.5, 0.5)

        e_der = (e_t - self.prev_e_heading) / delta_t if delta_t > 0 else 0.0
        
        omega = (self.kp * e_t) + \
                (self.ki * self.prev_int_heading) + \
                (self.kd * e_der)
        
        self.prev_e_heading = e_t

        return v_ref, omega

    def OffsetControl(self,
                      v_ref: float,
                      y_ref: float,
                      y_curr: float,
                      delta_t: float
                      ) -> Tuple[float, float]:
        """
        PID performing lateral offset control.
        Args:
            v_ref:      linear Duckiebot speed.
            y_ref:      reference heading pose.
            y_curr:     the current estimated "y" coordinate (offset)
            delta_t:    time interval since last call.
        Returns:
            v:          linear velocity of the Duckiebot
            omega:      angular velocity of the Duckiebot
        """
        e_t = y_ref - y_curr

        self.prev_int_offset += e_t * delta_t
        
        e_der = (e_t - self.prev_e_offset) / delta_t if delta_t > 0 else 0.0
        
        omega = (self.kp * e_t) + \
                (self.ki * self.prev_int_offset) + \
                (self.kd * e_der)
        
        self.prev_e_offset = e_t

        return v_ref, omega

    def SetGains(self, kp: float, ki: float, kd: float) -> None:
        # Set the PID gains
        self.kp = kp
        self.ki = ki
        self.kd = kd
        # Reset accum
        self.prev_int_offset = 0
        self.prev_int_heading = 0