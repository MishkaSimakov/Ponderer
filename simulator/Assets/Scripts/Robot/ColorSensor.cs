using UnityEngine;

// Looks along its own +Z. Reads the red channel of the surface texture, which
// the brick's REF-RAW difference matches byte for byte, and averages the cone.
public class ColorSensor : MonoBehaviour, IArenaResettable
{
    [SerializeField] LayerMask testPadMask;
    [SerializeField] bool drawRays;

    int rays = 21;

    // Full apex angle of the cone, degrees.
    float coneAngle = 40f;
    float range = 1f;

    // Per unit: texture byte to the reading of this physical sensor.
    float gain = 1f;
    float offset = 0f;
    // Measured reading with nothing under the sensor.
    float missValue = 0f;

    // Spins the whole spiral about the cone axis, radians.
    float phiOffset = 0f;

    // ratio between values in MODE_REF_RAW and values in MODE_COL_REFLECT
    float rawToReflectedCoef = 0.3f;

    const float GoldenAngle = 2.39996323f;

    public ResetPhase Phase { get { return ResetPhase.State; } }

    public void OnArenaReset(ArenaContext ctx)
    {
        // Domain randomization:
        // - coneAngle
        // - gain
        // - phiOffset
    }

    public float Reflected
    {
        get
        {
            float sum = 0f;
            for (int i = 0; i < rays; i++)
                sum += Sample(Direction(i));

            return Mathf.Clamp(Mathf.Round(rawToReflectedCoef * sum / rays), 0, 100);
        }
    }

    // Sunflower spread: even density over the cone for any ray count.
    Vector3 Direction(int i)
    {
        float theta = Mathf.Sqrt((i + 0.5f) / rays) * coneAngle * 0.5f * Mathf.Deg2Rad;
        float phi = i * GoldenAngle + phiOffset;
        float sin = Mathf.Sin(theta);

        return transform.TransformDirection(
            new Vector3(sin * Mathf.Cos(phi), sin * Mathf.Sin(phi), Mathf.Cos(theta)));
    }

    float Sample(Vector3 direction)
    {
        if (!Physics.Raycast(transform.position, direction, out RaycastHit hit, range, testPadMask))
        {
            Draw(transform.position + direction * range, Color.magenta);
            return missValue;
        }

        Material material = hit.transform.GetComponent<Renderer>().sharedMaterial;
        Vector2 uv = Vector2.Scale(hit.textureCoord, material.mainTextureScale)
                     + material.mainTextureOffset;
        float red = ((Texture2D)material.mainTexture).GetPixelBilinear(uv.x, uv.y).r;

        Draw(hit.point, new Color(red, red, red));

        return gain * 255f * red + offset;
    }

    // Grey is the value the ray sampled, magenta is a miss.
    void Draw(Vector3 end, Color color)
    {
        if (drawRays)
            Debug.DrawLine(transform.position, end, color);
    }
}
