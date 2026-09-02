using UnityEngine;

// Charged for every second the robot is off the line.
public class OffTrackPenalty : MonoBehaviour, IReward
{
    [SerializeField] float penaltyPerSecond = 1.25f;

    public float Evaluate(in RewardContext ctx)
    {
        return ctx.OffTrack ? -penaltyPerSecond * ctx.Dt : 0f;
    }
}
