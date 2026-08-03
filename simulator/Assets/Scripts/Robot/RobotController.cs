using System;
using UnityEngine;

public class RobotController : MonoBehaviour, IArenaResettable
{
    public const int ObsDim = 6;
    public const int ActionDim = 2;

    [SerializeField] UltrasonicSensor ultrasonic;
    [SerializeField] ColorSensor leftColor;
    [SerializeField] ColorSensor rightColor;

    [SerializeField] LargeMotor leftMotor;
    [SerializeField] LargeMotor rightMotor;

    // Brick output voltage. It depends on batteries.
    static readonly Vector2 VoltageRange = new Vector2(6f, 9f);
    public float Voltage { get; private set; }

    // The brick's own clock, seconds since the episode started.
    public float ElapsedSeconds { get; private set; }

    public ResetPhase Phase { get { return ResetPhase.State; } }

    public void OnArenaReset(ArenaContext ctx)
    {
        ElapsedSeconds = 0f;

        // Domain Randomization
        ArenaRandom rng = ctx.PhysicsRng(this);
        Voltage = rng.Range(VoltageRange);
    }

    public void SetDuty(float left, float right)
    {
        leftMotor.SetDuty(left);
        rightMotor.SetDuty(right);
    }

    // Called once per physics substep by Bridge, never from FixedUpdate: manual
    // simulation does not drive FixedUpdate, which keeps running on wall clock.
    public void Tick(float dt)
    {
        ElapsedSeconds += dt;
        leftMotor.Tick(dt);
        rightMotor.Tick(dt);
    }

    // Same order and units as server/main.py logs from the brick.
    public void Observe(float[] destination, int offset)
    {
        destination[offset + 0] = ElapsedSeconds;
        destination[offset + 1] = ultrasonic.DistanceCm;
        destination[offset + 2] = leftColor.Reflected;
        destination[offset + 3] = rightColor.Reflected;
        destination[offset + 4] = leftMotor.GetDegrees();
        destination[offset + 5] = rightMotor.GetDegrees();
    }
}
