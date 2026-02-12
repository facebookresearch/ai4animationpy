from ai4animation import FABRIK, Rotation, Vector3


class LegIK:
    def __init__(self, ankleIK: FABRIK, ballIK: FABRIK):
        self.AnkleIK = ankleIK
        self.BallIK = ballIK
        ankle_pos = ankleIK.LastBone().GetPosition().copy()
        ball_pos = ballIK.LastBone().GetPosition().copy()
        self.AnkleBaseline: float = ankle_pos[..., 1]
        self.BallBaseline: float = ball_pos[..., 1]
        self.AnkleBallDistance: float = Vector3.Distance(ankle_pos, ball_pos)

        self.AnkleTargetPosition = Vector3.Create(0, 0, 0)
        self.BallTargetPosition = Vector3.Create(0, 0, 0)
        self.AnkleTargetRotation = Rotation.Euler(0, 0, 0)
        self.BallTargetRotation = Rotation.Euler(0, 0, 0)

        self.ResetTargets()

    def Solve(
        self,
        ankleContact: float,
        ballContact: float,
        maxIterations: int,
        maxAccuracy: float,
    ):
        self.SolveAnkle(ankleContact, maxIterations, maxAccuracy)
        self.SolveBall(ballContact, maxIterations, maxAccuracy)

    def SolveAnkle(self, contact: float, maxIterations: int, maxAccuracy: float):
        locked_pos = self.AnkleTargetPosition.copy()
        locked_pos[..., 1] = max(
            Vector3.Lerp(locked_pos[..., 1], self.AnkleBaseline, contact),
            self.AnkleBaseline,
        )

        self.AnkleTargetPosition = Vector3.Lerp(
            self.AnkleIK.LastBone().GetPosition(), locked_pos, contact
        )

        self.AnkleTargetRotation = Rotation.Interpolate(
            self.AnkleIK.LastBone().GetRotation(),
            self.AnkleTargetRotation,
            0.5 * contact,
        )

        self.AnkleIK.Solve(
            self.AnkleTargetPosition,
            self.AnkleTargetRotation,
            maxIterations,
            maxAccuracy,
        )

    def SolveBall(self, contact: float, maxIterations: int, maxAccuracy: float):
        locked_pos = self.BallTargetPosition.copy()
        locked_pos[..., 1] = max(
            Vector3.Lerp(locked_pos[..., 1], self.BallBaseline, contact),
            self.BallBaseline,
        )

        self.BallTargetPosition = Vector3.Lerp(
            self.BallIK.LastBone().GetPosition(), locked_pos, contact
        )

        self.BallTargetPosition = self.AnkleTargetPosition + (
            self.AnkleBallDistance
            * Vector3.Normalize(self.BallTargetPosition - self.AnkleTargetPosition)
        )

        self.BallIK.Solve(self.BallTargetPosition, None, maxIterations, maxAccuracy)

    def ResetTargets(self):
        self.AnkleTargetPosition = self.AnkleIK.LastBone().GetPosition().copy()
        self.AnkleTargetRotation = self.AnkleIK.LastBone().GetRotation().copy()
        self.BallTargetPosition = self.BallIK.LastBone().GetPosition().copy()
        self.BallTargetRotation = self.BallIK.LastBone().GetRotation().copy()
