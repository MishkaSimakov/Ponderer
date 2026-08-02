using System;
using UnityEngine;

// Distance travelled along the centerline since the previous step, signed: going
// back down the track costs what going up it paid. Off the line it pays nothing,
// otherwise driving away from the line and back would still collect progress.
public class ProgressReward : MonoBehaviour, IReward, IArenaResettable
{
    [SerializeField] TrackController track;
    [SerializeField] RobotController robot;
    // Reward per meter of centerline.
    [SerializeField] float scale = 10f;

    float arc;

    public ResetPhase Phase { get { return ResetPhase.State; } }

    void Awake()
    {
        if (track == null) throw new Exception("ProgressReward.track is not set on " + name);
        if (robot == null) throw new Exception("ProgressReward.robot is not set on " + name);
    }

    public void OnArenaReset(ArenaContext ctx)
    {
        arc = track.Sample(robot.transform.position).Arc;
    }

    public float Evaluate(in RewardContext ctx)
    {
        float previous = arc;
        arc = ctx.Track.Arc;
        return ctx.OffTrack ? 0f : scale * (arc - previous);
    }
}
