// Evaluated once per control step, on the state the step ended in. A reward that
// needs history keeps it itself and implements IArenaResettable to clear it.
public interface IReward
{
    float Evaluate(in RewardContext ctx);
}
