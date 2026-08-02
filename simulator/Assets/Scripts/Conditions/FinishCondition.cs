using System;
using UnityEngine;

// The episode ends once the robot is within radius of the end of the line.
// Polled at the step boundary: the finish is a disc wider than one step of
// travel, so the robot cannot cross it unnoticed.
public class FinishCondition : MonoBehaviour, IEpisodeCondition
{
    [SerializeField] TrackController track;
    [SerializeField] RobotController robot;
    [SerializeField] float radius = 0.05f;
    [SerializeField] float reward = 10f;

    void Awake()
    {
        if (track == null) throw new Exception("FinishCondition.track is not set on " + name);
        if (robot == null) throw new Exception("FinishCondition.robot is not set on " + name);
    }

    public bool Terminated
    {
        get { return track.DistanceToFinish(robot.transform.position) <= radius; }
    }

    public float Reward { get { return reward; } }
}
