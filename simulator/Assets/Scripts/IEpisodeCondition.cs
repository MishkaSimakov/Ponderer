// Latched during physics substeps, read by Arena at the control step boundary.
// Latching rather than polling: triggers fire inside Physics.Simulate, and the
// robot can enter and leave a zone within a single control step.
public interface IEpisodeCondition
{
    bool Terminated { get; }
}
