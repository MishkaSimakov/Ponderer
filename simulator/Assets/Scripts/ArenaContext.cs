using System;
using System.Text;
using UnityEngine;

public class ArenaRandom
{
    readonly System.Random rng;
    readonly bool enabled;

    public ArenaRandom(int seed, bool enabled)
    {
        this.rng = new System.Random(seed & 0x7fffffff);
        this.enabled = enabled;
    }

    public bool Enabled { get { return enabled; } }

    public float Value { get { return (float)rng.NextDouble(); } }

    public int Int(int maxExclusive) { return rng.Next(maxExclusive); }

    // Returns the midpoint when this randomization category is disabled.
    public float Range(Vector2 range)
    {
        return enabled ? Mathf.Lerp(range.x, range.y, Value) : 0.5f * (range.x + range.y);
    }

    // Box-Muller. Returns zero when this randomization category is disabled.
    public float Normal(float sigma)
    {
        if (!enabled) return 0f;
        double u1 = 1.0 - rng.NextDouble();
        double u2 = rng.NextDouble();
        return sigma * (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }
}

// Every component draws from its own stream, derived from the episode seed and the
// component's position in the prefab. Adding or removing a randomized component
// therefore cannot change what any other component generates for a given seed.
public class ArenaContext
{
    const int ScenarioCategory = 0;
    const int PhysicsCategory = 1;

    readonly Transform root;
    readonly int seed;
    readonly bool scenario;
    readonly bool physics;

    public ArenaContext(Transform root, int seed, bool scenario, bool physics)
    {
        this.root = root;
        this.seed = seed;
        this.scenario = scenario;
        this.physics = physics;
    }

    public int Seed { get { return seed; } }

    public ArenaRandom ScenarioRng(Component c)
    {
        return new ArenaRandom(Mix(Mix(seed, StableId(c)), ScenarioCategory), scenario);
    }

    public ArenaRandom PhysicsRng(Component c)
    {
        return new ArenaRandom(Mix(Mix(seed, StableId(c)), PhysicsCategory), physics);
    }

    int StableId(Component c)
    {
        StringBuilder path = new StringBuilder();
        Transform t = c.transform;
        while (t != root)
        {
            if (t == null) throw new Exception(c.name + " is not under the arena root");
            path.Insert(0, t.name).Insert(0, '/');
            t = t.parent;
        }
        path.Append(':').Append(c.GetType().Name);
        return Fnv(path.ToString());
    }

    public static int DeriveSeed(int sessionSeed, int arena, int episode)
    {
        return Mix(Mix(sessionSeed, arena), episode);
    }

    static int Mix(int a, int b)
    {
        unchecked
        {
            uint h = 2166136261u;
            h = (h ^ (uint)a) * 16777619u;
            h = (h ^ (uint)b) * 16777619u;
            return (int)h;
        }
    }

    // Written out rather than using string.GetHashCode, which is not stable across runs.
    static int Fnv(string s)
    {
        unchecked
        {
            uint h = 2166136261u;
            for (int i = 0; i < s.Length; i++) h = (h ^ s[i]) * 16777619u;
            return (int)h;
        }
    }
}
