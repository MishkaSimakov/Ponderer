using System;
using UnityEngine;

// Places the robot on the start of the generated track with a randomized heading.
// Takes its spawn pose from the track, which is why this runs in the Placement
// phase, after World. The robot object only centers the model, so it goes onto
// the pad plane as is.
public class RobotPlacer : MonoBehaviour, IArenaResettable
{
    [SerializeField] Rigidbody body;
    [SerializeField] TrackController track;
    [SerializeField] Vector2 headingRange = new Vector2(-10f, 10f);

    public ResetPhase Phase { get { return ResetPhase.Placement; } }

    void Awake()
    {
        if (body == null) throw new Exception("RobotPlacer.body is not set on " + name);
        if (track == null) throw new Exception("RobotPlacer.track is not set on " + name);
    }

    public void OnArenaReset(ArenaContext ctx)
    {
        float heading = ctx.ScenarioRng(this).Range(headingRange);

        body.transform.SetPositionAndRotation(
            track.Start.position,
            track.Start.rotation * Quaternion.Euler(0f, heading, 0f)
        );
        body.linearVelocity = Vector3.zero;
        body.angularVelocity = Vector3.zero;
    }
}
