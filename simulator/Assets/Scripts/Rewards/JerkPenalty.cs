using UnityEngine;

// Charged while the commanded voltage of either motor changes faster than the
// threshold. The first step of an episode is measured against the standstill the
// robot resets to.
public class JerkPenalty : MonoBehaviour, IReward, IArenaResettable
{
    // Rate of voltage change tolerated for free, volts per second.
    [SerializeField] float thresholdVoltsPerSecond = 2.5f;
    [SerializeField] float penaltyPerSecond = 2.5f;

    RobotAction previous;

    public ResetPhase Phase { get { return ResetPhase.State; } }

    public void OnArenaReset(ArenaContext ctx)
    {
        previous = default;
    }

    public float Evaluate(in RewardContext ctx)
    {
        float tolerated = thresholdVoltsPerSecond * ctx.Dt;
        bool jerk = Mathf.Abs(ctx.Action.Left - previous.Left) > tolerated ||
            Mathf.Abs(ctx.Action.Right - previous.Right) > tolerated;
        previous = ctx.Action;
        return jerk ? -penaltyPerSecond * ctx.Dt : 0f;
    }
}
