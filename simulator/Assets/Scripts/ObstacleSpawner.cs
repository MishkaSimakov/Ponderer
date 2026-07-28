using System;
using UnityEngine;

// Stub: pooled children are toggled, never instantiated or destroyed. Destroy is
// deferred to end of frame, and physics runs inside Update, so a destroyed
// obstacle would still collide during the steps that follow in the same frame.
public class ObstacleSpawner : MonoBehaviour, IArenaResettable
{
    [SerializeField] Transform pool;
    [SerializeField] Vector2 countRange = new Vector2(0f, 3f);

    public ResetPhase Phase { get { return ResetPhase.World; } }

    void Awake()
    {
        if (pool == null) throw new Exception("ObstacleSpawner.pool is not set on " + name);
    }

    public void OnArenaReset(ArenaContext ctx)
    {
        ArenaRandom rng = ctx.ScenarioRng(this);
        int count = Mathf.RoundToInt(rng.Range(countRange));

        for (int i = 0; i < pool.childCount; i++)
            pool.GetChild(i).gameObject.SetActive(i < count);
    }
}
