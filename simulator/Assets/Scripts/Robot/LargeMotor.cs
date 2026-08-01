using UnityEngine;

// Motor doesn't rotate wheels. Instead, it applies a force directly to the point to which this MonoBehaviour is attached.
// The wheel is a velocity source: its cruise speed is the measured law alpha * |U| - beta, with separate constants per
// direction, and the contact force is linear in slip velocity and saturated by the friction circle.
// See the contact section of docs/motor_model.md.
public class LargeMotor : MonoBehaviour, IArenaResettable
{
    float wheelRadius = 0.0216f; // r
    float loadShare = 0.33f; // alpha_i
    float frictionCoef = 0.6f; // mu
    float longitudinalStiffness = 100f; // k_x
    float lateralStiffness = 100f; // k_y

    float forwardSpeedPerVolt = 0.04321f; // alpha, (m/s)/V at U > 0
    float forwardFrictionSpeed = 0.01539f; // beta, m/s at U > 0
    float reverseSpeedPerVolt = 0.04147f; // alpha, (m/s)/V at U < 0
    float reverseFrictionSpeed = 0.00977f; // beta, m/s at U < 0

    private Rigidbody body;
    private RobotController robotController;
    private float duty; // D_i, [-1, 1]

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
        angle = 0f;
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
        float angularVelocity = cruiseSpeed / wheelRadius;

        // Velocity of the body point under the contact patch, in wheel axes.
        Vector3 velocity = body.GetPointVelocity(transform.position);
        float alongSlip = angularVelocity * wheelRadius - Vector3.Dot(velocity, transform.forward);
        float sideSlip = -Vector3.Dot(velocity, transform.right);

        Vector2 force = new Vector2(longitudinalStiffness * alongSlip, lateralStiffness * sideSlip);
        force = Vector2.ClampMagnitude(force, frictionCoef * loadShare * body.mass * Physics.gravity.magnitude);

        angle += dt * angularVelocity;

        body.AddForceAtPosition(force.x * transform.forward + force.y * transform.right, transform.position);
    }
}
