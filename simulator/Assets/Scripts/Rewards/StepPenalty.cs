using UnityEngine;

// Fixed cost per second of episode: standing still is never free.
public class StepPenalty : MonoBehaviour, IReward
{
    [SerializeField] float penaltyPerSecond = 0.25f;

    public float Evaluate(in RewardContext ctx)
    {
        return -penaltyPerSecond * ctx.Dt;
    }
}
