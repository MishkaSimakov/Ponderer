using UnityEngine;

// Stub: generates nothing yet. Logs its draw so seed propagation is verifiable
// before there is any content to randomize.
public class TrackController : MonoBehaviour, IArenaResettable
{
    [SerializeField] Vector2 curvatureRange = new Vector2(0.2f, 1.0f);

    public ResetPhase Phase { get { return ResetPhase.World; } }

    public void OnArenaReset(ArenaContext ctx)
    {
        float curvature = ctx.ScenarioRng(this).Range(curvatureRange);
        Debug.Log(name + " track curvature " + curvature.ToString("F4"));
    }
}
