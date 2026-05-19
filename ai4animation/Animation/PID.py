# Copyright (c) Meta Platforms, Inc. and affiliates.
import numpy as np


class PID:
    def __init__(
        self,
        kp=1.0,
        ki=0.0,
        kd=0.0,
        setpoint=0.0,
        integral_limit=None,
        output_limit=None,
    ):
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd
        self.Setpoint = self._ToArray(setpoint)
        self.IntegralLimit = self._NormalizeLimit(integral_limit)
        self.OutputLimit = self._NormalizeLimit(output_limit)
        self.Reset()

    def Reset(self, integral=0.0, measurement=None):
        self.Integral = self._ToArray(integral)
        self.PreviousError = None
        self.PreviousMeasurement = (
            None if measurement is None else self._ToArray(measurement)
        )
        self.ProportionalTerm = None
        self.IntegralTerm = None
        self.DerivativeTerm = None
        self.FeedforwardTerm = None
        self.Output = None

    def Update(self, measurement, dt, setpoint=None, feedforward=0.0):
        if dt <= 0.0:
            return measurement
            raise ValueError("dt must be greater than zero")

        measurement = self._ToArray(measurement)
        if setpoint is not None:
            self.Setpoint = self._ToArray(setpoint)

        error = self.Setpoint - measurement
        integral = self._Clip(self.Integral + error * dt, self.IntegralLimit)

        if self.PreviousMeasurement is None:
            derivative = np.zeros_like(error)
        else:
            derivative = -(measurement - self.PreviousMeasurement) / dt

        feedforward = self._ToArray(feedforward)

        self.ProportionalTerm = self.Kp * error
        self.IntegralTerm = self.Ki * integral
        self.DerivativeTerm = self.Kd * derivative
        self.FeedforwardTerm = feedforward
        output = (
            self.ProportionalTerm
            + self.IntegralTerm
            + self.DerivativeTerm
            + self.FeedforwardTerm
        )
        output = self._Clip(output, self.OutputLimit)

        self.Integral = integral
        self.PreviousError = error
        self.PreviousMeasurement = measurement
        self.Output = output
        return self._ToNative(output)

    def __call__(self, measurement, dt, setpoint=None, feedforward=0.0):
        return self.Update(measurement, dt, setpoint, feedforward)

    def _Clip(self, value, limits):
        if limits is None:
            return value
        lower, upper = limits
        return np.clip(value, lower, upper)

    def _NormalizeLimit(self, value):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ValueError("limits must be None, a scalar, or a pair of bounds")
            return self._ToArray(value[0]), self._ToArray(value[1])

        bound = self._ToArray(value)
        return -bound, bound

    def _ToArray(self, value):
        return np.asarray(value, dtype=np.float64)

    def _ToNative(self, value):
        if np.shape(value) == ():
            return float(value)
        return np.array(value, copy=True)
