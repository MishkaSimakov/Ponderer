public readonly struct RewardContext
{
    public readonly RobotController Robot;

    // The action applied at the start of this step.
    public readonly RobotAction Action;

    // The centerline sampled under the robot at the end of this step.
    public readonly TrackSample Track;

    // Single definition of "off the line", shared by every reward.
    public readonly bool OffTrack;

    // Seconds this step covered. A term charged by the step rather than by the second
    // would pay the same for a step that lasted twice as long.
    public readonly float Dt;

    public RewardContext(RobotController robot, RobotAction action, TrackSample track,
        bool offTrack, float dt)
    {
        Robot = robot;
        Action = action;
        Track = track;
        OffTrack = offTrack;
        Dt = dt;
    }
}
