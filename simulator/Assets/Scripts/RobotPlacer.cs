using System;
using UnityEngine;

// Stub: places the robot at the arena origin with a randomized heading. The real
// version takes its spawn pose from the generated track, which is why this runs
// in the Placement phase, after World.
public class RobotPlacer : MonoBehaviour, IArenaResettable
{
    [SerializeField] Rigidbody body;
    [SerializeField] Vector2 headingRange = new Vector2(-10f, 10f);

    public ResetPhase Phase { get { return ResetPhase.Placement; } }

    void Awake()
    {
        if (body == null) throw new Exception("RobotPlacer.body is not set on " + name);
    }

    public void OnArenaReset(ArenaContext ctx)
    {
        float heading = ctx.ScenarioRng(this).Range(headingRange);

        body.position = transform.position;
        body.rotation = transform.rotation * Quaternion.Euler(0f, heading, 0f);
        body.linearVelocity = Vector3.zero;
        body.angularVelocity = Vector3.zero;
    }
}
