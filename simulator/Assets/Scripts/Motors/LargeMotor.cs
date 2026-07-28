using UnityEngine;

// Motor doesn't rotate wheels. Instead, it applies a force directly to the point to which this MonoBehaviour is attached.
public class LargeMotor : MonoBehaviour, IArenaResettable
{
    [SerializeField] float degreesPerSecondAtFullDuty = 900f;
    [SerializeField] float wheelRadius = 0.028f;
    [SerializeField] float loadShare = 0.4f;
    [SerializeField] float friction = 0.6f;
    [SerializeField] float longitudinalStiffness = 30f;
    [SerializeField] float lateralStiffness = 30f;

    private Rigidbody body;
    private float duty;
    public float Degrees { get; private set; }

    public ResetPhase Phase { get { return ResetPhase.State; } }

    void Awake()
    {
        body = GetComponentInParent<Rigidbody>();
    }

    public void OnArenaReset(ArenaContext ctx)
    {
        duty = 0f;
        Degrees = 0f;
    }

    public void SetDuty(float value)
    {
        duty = Mathf.Clamp(value, -100f, 100f);
    }

    // Duty commands wheel speed; the contact patch pulls the body towards it with a
    // force proportional to how much it slips, capped by friction against the load.
    public void Tick(float dt)
    {
        float degreesPerSecond = duty / 100f * degreesPerSecondAtFullDuty;
        Degrees += degreesPerSecond * dt;

        Vector3 velocity = body.GetPointVelocity(transform.position);
        float alongSlip = degreesPerSecond * Mathf.Deg2Rad * wheelRadius - Vector3.Dot(velocity, transform.forward);
        float sideSlip = -Vector3.Dot(velocity, transform.right);

        Vector2 force = new Vector2(longitudinalStiffness * alongSlip, lateralStiffness * sideSlip);
        force = Vector2.ClampMagnitude(force, friction * loadShare * body.mass * Physics.gravity.magnitude);

        body.AddForceAtPosition(force.x * transform.forward + force.y * transform.right, transform.position);
    }
}
