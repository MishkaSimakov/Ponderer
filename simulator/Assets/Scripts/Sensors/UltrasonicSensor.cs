using UnityEngine;

// Stub: constant reading. Real version raycasts a cone and models the NXT
// sensor's own measurement period, which is slower than the control rate.
public class UltrasonicSensor : MonoBehaviour, IArenaResettable
{
    public float DistanceCm { get; private set; }

    public ResetPhase Phase { get { return ResetPhase.State; } }

    public void OnArenaReset(ArenaContext ctx)
    {
        DistanceCm = 255f;
    }
}
