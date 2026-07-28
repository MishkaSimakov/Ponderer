using System;
using UnityEngine;

// Stub: integrates fake encoders from duty cycle so the loop is observable.
public class RobotController : MonoBehaviour, IArenaResettable
{
    public const int ObsDim = 5;
    public const int ActionDim = 2;

    [SerializeField] float degreesPerSecondAtFullDuty = 900f;

    [SerializeField] UltrasonicSensor ultrasonic;
    [SerializeField] ColorSensor leftColor;
    [SerializeField] ColorSensor rightColor;

    [SerializeField] LargeMotor leftMotor;
    [SerializeField] LargeMotor rightMotor;

    float leftDuty;
    float rightDuty;
    float leftDegrees;
    float rightDegrees;

    public ResetPhase Phase { get { return ResetPhase.State; } }

    public void OnArenaReset(ArenaContext ctx)
    {
        leftDuty = 0f;
        rightDuty = 0f;
        leftDegrees = 0f;
        rightDegrees = 0f;
    }

    public void SetDuty(float left, float right)
    {
        leftDuty = Mathf.Clamp(left, -100f, 100f);
        rightDuty = Mathf.Clamp(right, -100f, 100f);
    }

    // Called once per physics substep by Bridge, never from FixedUpdate: manual
    // simulation does not drive FixedUpdate, which keeps running on wall clock.
    public void Tick(float dt)
    {
        leftDegrees += leftDuty / 100f * degreesPerSecondAtFullDuty * dt;
        rightDegrees += rightDuty / 100f * degreesPerSecondAtFullDuty * dt;
    }

    // Same order and units as server/main.py logs from the brick.
    public void Observe(float[] destination, int offset)
    {
        destination[offset + 0] = ultrasonic.DistanceCm;
        destination[offset + 1] = 50f;
        destination[offset + 2] = 50f;
        destination[offset + 3] = Mathf.Round(leftDegrees);
        destination[offset + 4] = Mathf.Round(rightDegrees);
    }
}
