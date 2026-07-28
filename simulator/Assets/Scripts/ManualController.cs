using System;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.Controls;

// Keyboard replaces python: same stepping path as Bridge, clocked by real time.
// Episodes never truncate here, reset is manual.
public class ManualController : MonoBehaviour
{
    static readonly string[] ObsNames = { "distance", "left_color", "right_color", "left_position", "right_position" };

    [SerializeField] Arena arenaPrefab;
    [SerializeField] float controlPeriod = 0.05f;
    [SerializeField] float physicsDt = 0.005f;
    [SerializeField] int seed = 0;
    [SerializeField] bool randomize = false;

    Arena arena;
    int substeps;
    float accumulator;
    float level = 50f;
    float left;
    float right;
    readonly float[] obs = new float[RobotController.ObsDim];

    void Awake()
    {
        if (arenaPrefab == null) throw new Exception("ManualController.arenaPrefab is not set");

        Physics.simulationMode = SimulationMode.Script;
        Time.fixedDeltaTime = physicsDt;
        substeps = Mathf.RoundToInt(controlPeriod / physicsDt);

        arena = Instantiate(arenaPrefab, Vector3.zero, Quaternion.identity);
        arena.Initialize(0, seed);
        arena.ResetEpisode(randomize, randomize, arena.NextSeed());
    }

    void Update()
    {
        Keyboard kb = Keyboard.current;

        if (kb.rKey.wasPressedThisFrame) arena.ResetEpisode(randomize, randomize, arena.NextSeed());
        if (kb.equalsKey.wasPressedThisFrame || kb.numpadPlusKey.wasPressedThisFrame) level = Mathf.Min(level + 10f, 100f);
        if (kb.minusKey.wasPressedThisFrame || kb.numpadMinusKey.wasPressedThisFrame) level = Mathf.Max(level - 10f, 10f);

        float throttle = Axis(kb.wKey, kb.sKey);
        float steer = Axis(kb.dKey, kb.aKey);
        bool stop = kb.spaceKey.isPressed;
        left = stop ? 0f : Mathf.Clamp((throttle + steer) * level, -100f, 100f);
        right = stop ? 0f : Mathf.Clamp((throttle - steer) * level, -100f, 100f);

        // Clamped so a frame spike cannot make the loop chase an ever growing debt.
        accumulator = Mathf.Min(accumulator + Time.deltaTime, 4f * controlPeriod);
        while (accumulator >= controlPeriod)
        {
            accumulator -= controlPeriod;
            arena.ApplyAction(left, right);
            for (int s = 0; s < substeps; s++)
            {
                arena.Tick(physicsDt);
                Physics.Simulate(physicsDt);
            }
        }

        arena.Observe(obs, 0);
    }

    static float Axis(ButtonControl positive, ButtonControl negative)
    {
        return (positive.isPressed ? 1f : 0f) - (negative.isPressed ? 1f : 0f);
    }

    void OnGUI()
    {
        GUI.skin.label.fontSize = 14;
        GUILayout.BeginArea(new Rect(10f, 10f, 280f, 230f), GUI.skin.box);

        GUILayout.Label("duty   L " + left.ToString("F0") + "   R " + right.ToString("F0"));
        GUILayout.Label("level  " + level.ToString("F0"));
        GUILayout.Space(8f);
        for (int i = 0; i < ObsNames.Length; i++)
            GUILayout.Label(ObsNames[i] + "   " + obs[i].ToString("F1"));
        GUILayout.Space(8f);
        GUILayout.Label("WASD drive   -/+ level\nspace stop   R reset");

        GUILayout.EndArea();
    }
}
