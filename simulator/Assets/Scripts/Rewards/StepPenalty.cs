using UnityEngine;

// Fixed cost per control step: standing still is never free.
public class StepPenalty : MonoBehaviour, IReward
{
    [SerializeField] float penalty = 0.01f;

    public float Evaluate(in RewardContext ctx)
    {
        return -penalty;
    }
}
