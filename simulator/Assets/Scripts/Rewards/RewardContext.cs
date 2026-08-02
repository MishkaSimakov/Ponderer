public readonly struct RewardContext
{
    public readonly RobotController Robot;

    // The action applied at the start of this step.
    public readonly RobotAction Action;

    // The centerline sampled under the robot at the end of this step.
    public readonly TrackSample Track;

    // Single definition of "off the line", shared by every reward.
    public readonly bool OffTrack;

    public RewardContext(RobotController robot, RobotAction action, TrackSample track, bool offTrack)
    {
        Robot = robot;
        Action = action;
        Track = track;
        OffTrack = offTrack;
    }
}
