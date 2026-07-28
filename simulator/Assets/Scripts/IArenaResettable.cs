// World runs first (track, obstacles, surface), then Placement (robot pose, which
// depends on the generated world), then State (counters, filters, velocities).
public enum ResetPhase
{
    World = 0,
    Placement = 1,
    State = 2
}

public interface IArenaResettable
{
    ResetPhase Phase { get; }
    void OnArenaReset(ArenaContext ctx);
}
