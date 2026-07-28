using UnityEngine;

// Stub: samples but applies nothing. Uses PhysicsRng, so these draws are disabled
// when validating the simulator against a real log, while scenario randomization
// stays on.
public class PhysicsRandomizer : MonoBehaviour, IArenaResettable
{
    [SerializeField] Vector2 frictionRange = new Vector2(0.6f, 1.0f);
    [SerializeField] Vector2 massScaleRange = new Vector2(0.95f, 1.05f);

    public ResetPhase Phase { get { return ResetPhase.World; } }

    public void OnArenaReset(ArenaContext ctx)
    {
        ArenaRandom rng = ctx.PhysicsRng(this);
        float friction = rng.Range(frictionRange);
        float massScale = rng.Range(massScaleRange);
        Debug.Log(name + " friction " + friction.ToString("F4") + " mass scale " + massScale.ToString("F4"));
    }
}
