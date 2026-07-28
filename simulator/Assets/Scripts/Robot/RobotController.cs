using System;
using UnityEngine;

// Stub: integrates fake encoders from duty cycle so the loop is observable.
public class RobotController : MonoBehaviour, IArenaResettable
{
    public const int ObsDim = 5;
    public const int ActionDim = 2;

    [SerializeField] UltrasonicSensor ultrasonic;
    [SerializeField] ColorSensor leftColor;
    [SerializeField] ColorSensor rightColor;

    [SerializeField] LargeMotor leftMotor;
    [SerializeField] LargeMotor rightMotor;

    public ResetPhase Phase { get { return ResetPhase.State; } }

    public void OnArenaReset(ArenaContext ctx)
    {
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
        leftMotor.Tick(dt);
        rightMotor.Tick(dt);
    }

    // Same order and units as server/main.py logs from the brick.
    public void Observe(float[] destination, int offset)
    {
        destination[offset + 0] = ultrasonic.DistanceCm;
        destination[offset + 1] = 50f;
        destination[offset + 2] = 50f;
        destination[offset + 3] = leftMotor.Degrees;
        destination[offset + 4] = rightMotor.Degrees;
    }
}
