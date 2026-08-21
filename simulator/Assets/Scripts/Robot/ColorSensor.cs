using UnityEngine;

// Looks along its own +Z. Reads the red channel of the surface texture, which
// the brick's REF-RAW difference matches byte for byte, and averages the cone.
public class ColorSensor : MonoBehaviour, IArenaResettable
{
    [SerializeField] LayerMask testPadMask;
    [SerializeField] bool drawRays;

    int rays = 25;

    // Full apex angle of the cone, degrees.
    static readonly Vector2 ConeAngleRange = new Vector2(58f, 62f);
    float coneAngle;

    float range = 1f;

    // Per unit: texture byte to the reading of this physical sensor.
    static readonly Vector2 GainRange = new Vector2(0.99f, 1.01f);
    float gain;

    static readonly Vector2 OffsetRange = new Vector2(-1f, 1f);
    float offset;

    // Measured reading with nothing under the sensor.
    float missValue = 0f;

    // Spins the whole spiral about the cone axis, radians.
    float phiOffset;

    // ratio between values in MODE_REF_RAW and values in MODE_COL_REFLECT
    float rawToReflectedCoef = 0.3f;

    const float GoldenAngle = 2.39996323f;

    // True while drawing from OnDrawGizmos, where Debug.DrawLine is a no-op.
    bool gizmoPass;

    public ResetPhase Phase { get { return ResetPhase.State; } }

    public void OnArenaReset(ArenaContext ctx)
    {
        // Domain Randomization
        ArenaRandom rng = ctx.PhysicsRng(this);

        coneAngle = rng.Range(ConeAngleRange);
        gain = rng.Range(GainRange);
        offset = rng.Range(OffsetRange);
        phiOffset = rng.Range(new Vector2(0, Mathf.PI * 2));
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

        return transform.TransformDirection(
            new Vector3(
                Mathf.Sin(theta) * Mathf.Cos(phi),
                Mathf.Sin(theta) * Mathf.Sin(phi),
                Mathf.Cos(theta)
            ));
    }

    float Sample(Vector3 direction)
    {
        if (!Physics.Raycast(transform.position, direction, out RaycastHit hit, range, testPadMask))
        {
            Draw(transform.position + direction * range, Color.magenta);
            return missValue;
        }

        if (!TryReadRed(hit, out float red))
        {
            Draw(hit.point, Color.cyan);
            return missValue;
        }

        Draw(hit.point, new Color(red, red, red));

        return gain * 255f * red + offset;
    }

    // Cyan marks a hit the sensor could not read: no renderer, no readable
    // Texture2D. In play mode this would have thrown; in the editor you can be
    // pointed at anything, so it degrades to a miss instead.
    bool TryReadRed(RaycastHit hit, out float red)
    {
        red = 0f;

        Renderer renderer = hit.transform.GetComponent<Renderer>();
        if (renderer == null)
            return false;

        Material material = renderer.sharedMaterial;
        if (material == null || !(material.mainTexture is Texture2D texture))
            return false;

        if (!texture.isReadable)
            return false;

        Vector2 uv = Vector2.Scale(hit.textureCoord, material.mainTextureScale)
                     + material.mainTextureOffset;
        red = texture.GetPixelBilinear(uv.x, uv.y).r;
        return true;
    }

    // Grey is the value the ray sampled, magenta is a miss.
    void Draw(Vector3 end, Color color)
    {
        if (!drawRays)
            return;

        if (gizmoPass)
        {
            Gizmos.color = color;
            Gizmos.DrawLine(transform.position, end);
            Gizmos.DrawSphere(end, 0.004f);
        }
        else
        {
            Debug.DrawLine(transform.position, end, color);
        }
    }

    // Scene view preview. Stripped from builds; Unity never calls it there.
    void OnDrawGizmos()
    {
        if (!drawRays)
            return;

        // OnArenaReset has not run outside play mode, so coneAngle would be 0
        // and the whole cone would collapse into a single ray along +Z.
        if (!Application.isPlaying)
        {
            coneAngle = (ConeAngleRange.x + ConeAngleRange.y) * 0.5f;
            gain = 1f;
            offset = 0f;
            phiOffset = 0f;
        }

        // Queries otherwise use collider positions from the last physics sync,
        // which lag behind while you drag things around the Scene view.
        Physics.SyncTransforms();

        gizmoPass = true;
        for (int i = 0; i < rays; i++)
            Sample(Direction(i));
        gizmoPass = false;
    }
}