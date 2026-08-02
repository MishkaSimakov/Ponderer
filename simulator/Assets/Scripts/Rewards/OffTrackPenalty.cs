using UnityEngine;

// Charged every step the robot is off the line.
public class OffTrackPenalty : MonoBehaviour, IReward
{
    [SerializeField] float penalty = 0.5f;

    public float Evaluate(in RewardContext ctx)
    {
        return ctx.OffTrack ? -penalty : 0f;
    }
}
