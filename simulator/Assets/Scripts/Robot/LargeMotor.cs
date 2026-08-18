using UnityEngine;

// Motor doesn't rotate wheels. Instead, it applies a force directly to the point to which this MonoBehaviour is attached.
// The wheel is a velocity source: it approaches the measured cruise law alpha * |U| - beta, with separate constants per
// direction, as a first order lag with time constant tau. The contact force is linear in slip velocity, saturated by
// the friction circle and by what one step can absorb without reversing the slip.
// See the contact section of docs/motor_model.md.
public class LargeMotor : MonoBehaviour, IArenaResettable
{
    float wheelRadius = 0.0216f; // r
    float loadShare = 0.33f; // alpha_i
    float frictionCoef = 0.6f; // mu
    float longitudinalStiffness = 100f; // k_x
    float lateralStiffness = 100f; // k_y

    static readonly Vector2 SpeedTauRange = new Vector2(0.05f, 0.12f);
    float speedTau; // tau, s to close 63% of the gap to cruise speed

    static readonly Vector2 ForwardSpeedPerVoltRange = new Vector2(0.042f, 0.045f);
    static readonly Vector2 ForwardFrictionSpeedRange = new Vector2(0.01f, 0.03f);
    static readonly Vector2 ReverseSpeedPerVoltRange = new Vector2(0.040f, 0.043f);
    static readonly Vector2 ReverseFrictionSpeedRange = new Vector2(0.006f, 0.010f);

    float forwardSpeedPerVolt; // alpha, (m/s)/V at U > 0
    float forwardFrictionSpeed; // beta, m/s at U > 0
    float reverseSpeedPerVolt; // alpha, (m/s)/V at U < 0
    float reverseFrictionSpeed; // beta, m/s at U < 0

    private Rigidbody body;
    private RobotController robotController;
    private float duty; // D_i, [-1, 1]
    private float speed; // v_i, m/s at the contact patch, lagging cruise speed

    private float angle; // theta_i, rad

    public ResetPhase Phase { get { return ResetPhase.State; } }

    void Awake()
    {
        body = GetComponentInParent<Rigidbody>();
        robotController = GetComponentInParent<RobotController>();
    }

    public void OnArenaReset(ArenaContext ctx)
    {
        duty = 0f;
        speed = 0f;
        angle = 0f;

        // Domain Randomization
        ArenaRandom rng = ctx.PhysicsRng(this);

        speedTau = rng.Range(SpeedTauRange);

        forwardSpeedPerVolt = rng.Range(ForwardSpeedPerVoltRange);
        forwardFrictionSpeed = rng.Range(ForwardFrictionSpeedRange);
        reverseSpeedPerVolt = rng.Range(ReverseSpeedPerVoltRange);
        reverseFrictionSpeed = rng.Range(ReverseFrictionSpeedRange);
    }

    public void SetDuty(float value)
    {
        duty = Mathf.Clamp(value, -100f, 100f) / 100f;
    }

    public float GetDegrees()
    {
        return angle * Mathf.Rad2Deg;
    }

    public void Tick(float dt)
    {
        float voltage = robotController.Voltage * duty;
        float speedPerVolt = voltage >= 0f ? forwardSpeedPerVolt : reverseSpeedPerVolt;
        float frictionSpeed = voltage >= 0f ? forwardFrictionSpeed : reverseFrictionSpeed;

        // Zero inside the dead zone |U| < beta / alpha, where friction wins.
        float cruiseSpeed = Mathf.Max(speedPerVolt * Mathf.Abs(voltage) - frictionSpeed, 0f)
                            * Mathf.Sign(voltage);
        speed += (cruiseSpeed - speed) * (1f - Mathf.Exp(-dt / speedTau));
        float angularVelocity = speed / wheelRadius;

        // Velocity of the body point under the contact patch, in wheel axes.
        Vector3 velocity = body.GetPointVelocity(transform.position);

        float alongSlip = angularVelocity * wheelRadius - Vector3.Dot(velocity, transform.forward);
        float sideSlip = -Vector3.Dot(velocity, transform.right);

        Vector2 slip = new Vector2(alongSlip, sideSlip);
        Vector2 force = new Vector2(longitudinalStiffness * slip.x, lateralStiffness * slip.y);
        Vector2 direction = force.normalized;

        float friction = frictionCoef * loadShare * body.mass * Physics.gravity.magnitude;
        // Stepping is explicit, so the force must not reverse the slip within dt. Halved
        // because both wheels push in the same step, each sizing its limit as if alone.
        float stable = 0.5f * EffectiveMass(direction.x * transform.forward + direction.y * transform.right)
                       * Vector2.Dot(slip, direction) / dt;

//        Debug.Log($"v = {velocity}, w = {angularVelocity}, |F| = {force.magnitude}, |F_fr| = {friction}, |F_st| = {stable}, slip = {alongSlip} x {sideSlip}");

        force = Vector2.ClampMagnitude(force, Mathf.Min(friction, stable));

        angle += dt * angularVelocity;

        body.AddForceAtPosition(force.x * transform.forward + force.y * transform.right, transform.position);
    }

    // Mass the body shows to an impulse applied at the contact point along direction:
    // 1 / m_eff = 1 / M + (rho x d)^T I^-1 (rho x d), with I diagonal in its principal frame.
    float EffectiveMass(Vector3 direction)
    {
        Vector3 axis = Vector3.Cross(transform.position - body.worldCenterOfMass, direction);
        Vector3 local = Quaternion.Inverse(body.rotation * body.inertiaTensorRotation) * axis;
        Vector3 inertia = body.inertiaTensor;

        return 1f / (1f / body.mass
                     + Compliance(local.x, inertia.x)
                     + Compliance(local.y, inertia.y)
                     + Compliance(local.z, inertia.z));
    }

    // A frozen rotation axis reads back as zero inertia, meaning infinite: it yields nothing.
    static float Compliance(float lever, float inertia)
    {
        return inertia > 0f ? lever * lever / inertia : 0f;
    }
}
