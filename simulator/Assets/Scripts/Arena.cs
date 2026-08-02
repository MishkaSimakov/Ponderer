using System;
using System.Linq;
using System.Text;
using UnityEngine;

public class Arena : MonoBehaviour
{
    [SerializeField] int maxSteps = 200;
    [SerializeField] RobotController robot;
    [SerializeField] TrackController track;
    // Lateral distance past which the robot counts as off the line, meters.
    [SerializeField] float offTrackDistance = 0.06f;
    [SerializeField] bool logRewards;

    readonly StringBuilder log = new StringBuilder();
    IArenaResettable[] resettables;
    IEpisodeCondition[] conditions;
    IReward[] rewards;
    readonly float[] terminalObs = new float[RobotController.ObsDim];
    RobotAction action;
    float episodeReward;
    int index;
    int sessionSeed;
    bool initialized;
    bool scenario;
    bool physics;

    public int Episode { get; private set; }
    public int Step { get; private set; }
    public bool Terminated { get; private set; }
    public bool Truncated { get; private set; }
    public float Reward { get; private set; }

    void Awake()
    {
        if (robot == null) throw new Exception("Arena.robot is not set on " + name);
        if (track == null) throw new Exception("Arena.track is not set on " + name);

        // Push based discovery: a component participates because it implements the
        // interface, not because it registered itself somewhere.
        resettables = GetComponentsInChildren<IArenaResettable>(true)
            .OrderBy(r => r.Phase)
            .ToArray();
        conditions = GetComponentsInChildren<IEpisodeCondition>(true).ToArray();
        rewards = GetComponentsInChildren<IReward>(true).ToArray();
    }

    // Called on handshake: the session seed comes from python, not the command line.
    public void Initialize(int index, int sessionSeed)
    {
        this.index = index;
        this.sessionSeed = sessionSeed;
        this.initialized = true;
        Episode = -1;
    }

    public void ResetEpisode(bool scenario, bool physics, int seed)
    {
        if (!initialized) throw new Exception("arena reset before handshake");
        this.scenario = scenario;
        this.physics = physics;

        Episode++;
        Step = 0;
        Terminated = false;
        Truncated = false;
        Reward = 0f;
        action = default;
        episodeReward = 0f;

        ArenaContext ctx = new ArenaContext(transform, seed, scenario, physics);
        for (int i = 0; i < resettables.Length; i++) resettables[i].OnArenaReset(ctx);

        // Transform writes are not visible to raycasts until physics is synced,
        // and reset must not simulate.
        Physics.SyncTransforms();
    }

    // Manual play has no step limit: only conditions end an episode.
    public void DisableTruncation()
    {
        maxSteps = 0;
    }

    public int NextSeed()
    {
        return ArenaContext.DeriveSeed(sessionSeed, index, Episode + 1);
    }

    public void ApplyAction(float left, float right)
    {
        action = new RobotAction(left, right);
        robot.SetDuty(left, right);
    }

    // Used instead of FixedUpdate.
    public void Tick(float dt)
    {
        robot.Tick(dt);
    }

    public void AdvanceStep()
    {
        Step++;

        // Summed before the auto reset, on the same state terminalObs captures.
        TrackSample sample = track.Sample(robot.transform.position);
        RewardContext ctx = new RewardContext(
            robot, action, sample, Mathf.Abs(sample.Offset) > offTrackDistance);
        float reward = 0f;
        for (int i = 0; i < rewards.Length; i++)
        {
            // A term is switched off by disabling its component in the inspector.
            if (!((Behaviour)rewards[i]).isActiveAndEnabled) continue;

            float value = rewards[i].Evaluate(in ctx);
            reward += value;
            Trace(rewards[i], value);
        }

        // Terminated: the episode genuinely ended, no future return exists.
        // Truncated: the step limit cut off an episode that would have continued.
        // A trainer bootstraps the value function for the second case only.
        bool terminated = false;
        for (int i = 0; i < conditions.Length; i++)
        {
            if (!conditions[i].Terminated) continue;
            terminated = true;
            reward += conditions[i].Reward;
            Trace(conditions[i], conditions[i].Reward);
        }
        bool truncated = !terminated && maxSteps > 0 && Step >= maxSteps;

        episodeReward += reward;

        // Before the auto reset, while Episode and Step still describe this step.
        Flush(reward);

        if (terminated || truncated)
        {
            FlushEpisode(terminated);
            robot.Observe(terminalObs, 0);
            ResetEpisode(scenario, physics, NextSeed());
        }

        Reward = reward;
        Terminated = terminated;
        Truncated = truncated;
    }

    void Trace(object source, float value)
    {
        if (!logRewards || Mathf.Abs(value) < 0.001f) return;
        if (log.Length > 0) log.Append(", ");
        log.Append(source.GetType().Name).Append(' ').Append(value.ToString("F4"));
    }

    void Flush(float total)
    {
        if (log.Length == 0) return;
        Debug.Log("arena " + index + " ep " + Episode + " step " + Step +
            ": " + log + " | total " + total.ToString("F4"));
        log.Length = 0;
    }

    void FlushEpisode(bool terminated)
    {
        if (!logRewards) return;
        Debug.Log("arena " + index + " ep " + Episode + " " +
            (terminated ? "terminated" : "truncated") + " after " + Step +
            " steps | episode total " + episodeReward.ToString("F4"));
    }

    public void Observe(float[] destination, int offset)
    {
        robot.Observe(destination, offset);
    }

    public void ObserveTerminal(float[] destination, int offset)
    {
        Array.Copy(terminalObs, 0, destination, offset, terminalObs.Length);
    }
}
