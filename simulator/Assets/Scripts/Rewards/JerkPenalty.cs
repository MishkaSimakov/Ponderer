using UnityEngine;

// Charged when the commanded voltage of either motor jumps by more than the
// threshold since the previous step. The first step of an episode is measured
// against the standstill the robot resets to.
public class JerkPenalty : MonoBehaviour, IReward, IArenaResettable
{
    // Voltage change tolerated for free, volts.
    [SerializeField] float threshold = 0.7f;
    [SerializeField] float penalty = 0.1f;

    RobotAction previous;

    public ResetPhase Phase { get { return ResetPhase.State; } }

    public void OnArenaReset(ArenaContext ctx)
    {
        previous = default;
    }

    public float Evaluate(in RewardContext ctx)
    {
        bool jerk = Mathf.Abs(ctx.Action.Left - previous.Left) > threshold ||
            Mathf.Abs(ctx.Action.Right - previous.Right) > threshold;
        previous = ctx.Action;
        return jerk ? -penalty : 0f;
    }
}
