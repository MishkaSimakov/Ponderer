public readonly struct RewardContext
{
    public readonly RobotController Robot;

    // The action applied at the start of this step.
    public readonly RobotAction Action;

    public RewardContext(RobotController robot, RobotAction action)
    {
        Robot = robot;
        Action = action;
    }
}
