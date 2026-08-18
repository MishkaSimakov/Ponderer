using System;

// Per arena values travel flat so Unity's JsonUtility can serialize them:
// actions is [l0, r0, l1, r1, ...], obs is arenas * obs_dim.
[Serializable]
public class Request
{
    public string cmd;
    public int version;
    public int session_seed;
    public int[] seeds;
    public bool randomize_scenario;
    public bool randomize_physics;
    public float[] actions;
}

[Serializable]
public class HandshakeResponse
{
    public int version;
    public int arenas;
    public float dt;
    public int obs_dim;
    public int action_dim;
    public string[] reward_terms;
}

[Serializable]
public class StateResponse
{
    public float[] obs;
    public float[] terminal_obs;
    public float[] reward;
    // Flat, arenas * reward_terms: what each term contributed to reward.
    public float[] terms;
    public bool[] terminated;
    public bool[] truncated;
    public int[] episode;
    public int[] step;
}
