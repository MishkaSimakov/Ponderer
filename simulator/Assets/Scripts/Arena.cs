using System;
using System.Linq;
using UnityEngine;

public class Arena : MonoBehaviour
{
    [SerializeField] int maxSteps = 200;
    [SerializeField] RobotController robot;

    IArenaResettable[] resettables;
    IEpisodeCondition[] conditions;
    readonly float[] terminalObs = new float[RobotController.ObsDim];
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

        // Push based discovery: a component participates because it implements the
        // interface, not because it registered itself somewhere.
        resettables = GetComponentsInChildren<IArenaResettable>(true)
            .OrderBy(r => r.Phase)
            .ToArray();
        conditions = GetComponentsInChildren<IEpisodeCondition>(true).ToArray();
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

        ArenaContext ctx = new ArenaContext(transform, seed, scenario, physics);
        for (int i = 0; i < resettables.Length; i++) resettables[i].OnArenaReset(ctx);

        // Transform writes are not visible to raycasts until physics is synced,
        // and reset must not simulate.
        Physics.SyncTransforms();
    }

    public int NextSeed()
    {
        return ArenaContext.DeriveSeed(sessionSeed, index, Episode + 1);
    }

    public void ApplyAction(float left, float right)
    {
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
        Reward = 0f;

        // Terminated: the episode genuinely ended, no future return exists.
        // Truncated: the step limit cut off an episode that would have continued.
        // A trainer bootstraps the value function for the second case only.
        bool terminated = false;
        for (int i = 0; i < conditions.Length; i++) terminated |= conditions[i].Terminated;
        bool truncated = !terminated && Step >= maxSteps;

        if (terminated || truncated)
        {
            robot.Observe(terminalObs, 0);
            ResetEpisode(scenario, physics, NextSeed());
        }

        Terminated = terminated;
        Truncated = truncated;
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
