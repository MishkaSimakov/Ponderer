using System;
using System.Linq;
using System.Text;
using UnityEngine;

public class Arena : MonoBehaviour
{
    [SerializeField] float maxSeconds = 20f;
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
    float[] termScratch;
    float[] terms;
    RobotAction action;
    float elapsed;
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

    // One name per reward term, rewards first then conditions; Terms holds their
    // values for the step Reward describes.
    public string[] TermNames { get; private set; }
    public float[] Terms { get { return terms; } }

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

        TermNames = rewards.Select(r => r.GetType().Name)
            .Concat(conditions.Select(c => c.GetType().Name))
            .ToArray();
        if (TermNames.Distinct().Count() != TermNames.Length)
            throw new Exception("two reward terms of the same type on " + name);
        termScratch = new float[TermNames.Length];
        terms = new float[TermNames.Length];
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
        elapsed = 0f;
        Terminated = false;
        Truncated = false;
        Reward = 0f;
        Array.Clear(terms, 0, terms.Length);
        action = default;
        episodeReward = 0f;

        ArenaContext ctx = new ArenaContext(transform, seed, scenario, physics);
        for (int i = 0; i < resettables.Length; i++)
        {
            // Switched off in the inspector: its Awake never ran, so it must not run.
            if (!((Behaviour)resettables[i]).isActiveAndEnabled) continue;
            resettables[i].OnArenaReset(ctx);
        }

        // Transform writes are not visible to raycasts until physics is synced,
        // and reset must not simulate.
        Physics.SyncTransforms();
    }

    // Manual play has no time limit: only conditions end an episode.
    public void DisableTruncation()
    {
        maxSeconds = 0f;
    }

    public int NextSeed()
    {
        return ArenaContext.DeriveSeed(sessionSeed, index, Episode + 1);
    }

    public void ApplyAction(float left, float right)
    {
        action = new RobotAction(left, right);
        robot.SetVolts(left, right);
    }

    // Used instead of FixedUpdate.
    public void Tick(float dt)
    {
        robot.Tick(dt);
    }

    public void AdvanceStep(float dt)
    {
        Step++;
        elapsed += dt;

        // Summed before the auto reset, on the same state terminalObs captures.
        TrackSample sample = track.Sample(robot.transform.position);
        RewardContext ctx = new RewardContext(
            robot, action, sample, Mathf.Abs(sample.Offset) > offTrackDistance, dt);
        float reward = 0f;
        Array.Clear(termScratch, 0, termScratch.Length);
        for (int i = 0; i < rewards.Length; i++)
        {
            // A term is switched off by disabling its component in the inspector.
            if (!((Behaviour)rewards[i]).isActiveAndEnabled) continue;

            float value = rewards[i].Evaluate(in ctx);
            reward += value;
            termScratch[i] = value;
            Trace(rewards[i], value);
        }

        // Terminated: the episode genuinely ended, no future return exists.
        // Truncated: the time limit cut off an episode that would have continued.
        // A trainer bootstraps the value function for the second case only.
        bool terminated = false;
        for (int i = 0; i < conditions.Length; i++)
        {
            if (!conditions[i].Terminated) continue;
            terminated = true;
            reward += conditions[i].Reward;
            termScratch[rewards.Length + i] = conditions[i].Reward;
            Trace(conditions[i], conditions[i].Reward);
        }
        bool truncated = !terminated && maxSeconds > 0f && elapsed >= maxSeconds;

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
        Array.Copy(termScratch, terms, terms.Length);
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
