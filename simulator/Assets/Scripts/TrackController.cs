using System;
using UnityEngine;

public struct TrackSample
{
    // Closest point of the centerline and the travel direction there, world.
    public Vector3 Position;
    public Vector3 Direction;
    // Lateral distance to the centerline, positive is left of travel, meters.
    public float Offset;
    // Distance from the start along the centerline, meters.
    public float Arc;
}

// Lives on the test pad. Draws the line into a runtime texture, which is what the
// color sensors read; the centerline polyline stays as ground truth for rewards
// and for placing the robot. Track coordinates are meters in the pad plane,
// measured from the center of the printed area: x runs start to finish, y is
// lateral.
public class TrackController : MonoBehaviour, IArenaResettable
{
    // Rectangle in track coordinates: axis is the long side, half its extents.
    struct Blotch
    {
        public Vector2 Center;
        public Vector2 Axis;
        public Vector2 Half;
        public Color32 Color;
    }

    // Printed area, meters. Centered on the pad, which may be larger.
    [SerializeField] Vector2 size = new Vector2(0.8f, 0.8f);
    [SerializeField] int resolution = 512;
    [SerializeField] Color lineColor = Color.black;
    [SerializeField] Color backgroundColor = Color.white;

    static readonly Vector2 LineWidthRange = new Vector2(0.024f, 0.026f);
    float lineWidth;

    // 1/m: the tightest turn the line is allowed to take.
    [SerializeField] float maxCurvature = 5f;
    [SerializeField] float margin = 0.05f;
    // Meters between the curvature control points the line interpolates through.
    [SerializeField] float curvatureLength = 0.25f;

    // Colored patches painted over the line, as on the real pad.
    const int MaxBlotches = 5;
    static readonly Vector2 BlotchSizeRange = new Vector2(0.01f, 0.08f);
    static readonly Vector2 BlotchSaturationRange = new Vector2(0.7f, 1f);
    static readonly Vector2 BlotchValueRange = new Vector2(0.6f, 1f);

    const float Step = 0.01f;
    // Bounding the heading keeps the line a graph over the track x axis, so it
    // cannot cross itself and start-to-finish stays unambiguous.
    const float MaxHeading = 60f * Mathf.Deg2Rad;

    Vector3 scale;
    Vector2 texel;
    // Fraction of the half width past which the line is forced back to the middle.
    float bendBand;

    Texture2D texture;
    Color32[] pixels;
    byte[] coverage;
    Vector2[] points;
    int count;
    Blotch[] blotches;
    int blotchCount;

    public ResetPhase Phase { get { return ResetPhase.World; } }

    public float LineWidth { get { return lineWidth; } }
    public float Length { get { return (count - 1) * Step; } }
    public Vector3 Finish { get { return ToWorld(points[count - 1]); } }

    public Pose Start
    {
        get
        {
            Vector3 forward = ToWorldDirection((points[1] - points[0]).normalized);
            return new Pose(ToWorld(points[0]), Quaternion.LookRotation(forward, transform.up));
        }
    }

    void Awake()
    {
        scale = transform.lossyScale;
        texel = size / resolution;

        // Lateral room a full-heading turn needs to straighten out again.
        float room = (1f - Mathf.Cos(MaxHeading)) / maxCurvature;
        float halfWidth = 0.5f * size.y - margin;
        if (halfWidth <= 2f * room)
            throw new Exception(name + ": maxCurvature too small to keep the line inside the area");
        bendBand = 1f - 2f * room / halfWidth;

        points = new Vector2[Mathf.CeilToInt(size.x / (Mathf.Cos(MaxHeading) * Step)) + 2];
        blotches = new Blotch[MaxBlotches];
        pixels = new Color32[resolution * resolution];
        coverage = new byte[resolution * resolution];
        texture = new Texture2D(resolution, resolution, TextureFormat.RGB24, false);
        texture.wrapMode = TextureWrapMode.Clamp;

        // Instancing the material: every arena draws its own track. The texture
        // covers only the printed area, the rest of the pad clamps to its border.
        Mesh mesh = GetComponent<MeshFilter>().sharedMesh;
        Vector2 u = UvAxis(mesh, 0, scale.x);
        Vector2 v = UvAxis(mesh, 2, scale.z);
        Vector2 tiling = new Vector2(1f / (u.x * size.x), 1f / (v.x * size.y));

        Material material = GetComponent<Renderer>().material;
        material.mainTexture = texture;
        material.mainTextureScale = tiling;
        material.mainTextureOffset = new Vector2(0.5f - u.y * tiling.x, 0.5f - v.y * tiling.y);
    }

    // Fits uv = slope * meters + intercept along one pad axis, meters measured from
    // the pad center. Read off the mesh: which way the pad's uv run is a property of
    // the mesh, and getting it backwards would mirror the texture off the polyline.
    static Vector2 UvAxis(Mesh mesh, int axis, float scale)
    {
        Vector3[] vertices = mesh.vertices;
        Vector2[] uv = mesh.uv;

        int min = 0;
        int max = 0;
        for (int i = 1; i < vertices.Length; i++)
        {
            if (vertices[i][axis] < vertices[min][axis]) min = i;
            if (vertices[i][axis] > vertices[max][axis]) max = i;
        }

        int channel = axis == 0 ? 0 : 1;
        float span = (vertices[max][axis] - vertices[min][axis]) * scale;
        float slope = (uv[max][channel] - uv[min][channel]) / span;
        if (slope == 0f) throw new Exception(mesh.name + ": uv do not follow the pad axes");

        return new Vector2(slope, uv[min][channel] - slope * vertices[min][axis] * scale);
    }

    public void OnArenaReset(ArenaContext ctx)
    {
        // Domain Randomization
        ArenaRandom rng = ctx.PhysicsRng(this);
        lineWidth = rng.Range(LineWidthRange);

        ArenaRandom scenario = ctx.ScenarioRng(this);
        Generate(scenario);
        GenerateBlotches(scenario);
        Rasterize();
    }

    // Integrates a heading driven by curvature interpolated between random control
    // points. With randomization off every draw is the midpoint of its symmetric
    // range, which is a straight line down the middle of the area.
    void Generate(ArenaRandom rng)
    {
        float halfWidth = 0.5f * size.y - margin;
        float endX = 0.5f * size.x - margin;

        Vector2 p = new Vector2(
            -endX, rng.Range(new Vector2(-0.5f * halfWidth, 0.5f * halfWidth)));
        float heading = rng.Range(new Vector2(-0.5f * MaxHeading, 0.5f * MaxHeading));

        float from = Curvature(rng);
        float to = Curvature(rng);
        float held = 0f;

        count = 0;
        points[count++] = p;
        while (p.x < endX && count < points.Length)
        {
            float u = held / curvatureLength;
            float curvature = Mathf.Lerp(from, to, u * u * (3f - 2f * u));

            float side = p.y / halfWidth;
            float bend = Mathf.InverseLerp(bendBand, 1f, Mathf.Abs(side));
            if (bend > 0f)
                curvature = Mathf.Lerp(curvature, -Mathf.Sign(side) * maxCurvature, bend);

            heading = Mathf.Clamp(heading + curvature * Step, -MaxHeading, MaxHeading);
            p += new Vector2(Mathf.Cos(heading), Mathf.Sin(heading)) * Step;
            points[count++] = p;

            held += Step;
            if (held >= curvatureLength)
            {
                held -= curvatureLength;
                from = to;
                to = Curvature(rng);
            }
        }
    }

    // Placed anywhere along the line, offset within the line width and turned by an
    // arbitrary angle. Nothing is drawn when scenario randomization is off.
    void GenerateBlotches(ArenaRandom rng)
    {
        blotchCount = rng.Enabled ? rng.Int(MaxBlotches + 1) : 0;

        for (int i = 0; i < blotchCount; i++)
        {
            float at = rng.Value * (count - 1);
            int segment = Mathf.Min((int)at, count - 2);

            Vector2 direction = (points[segment + 1] - points[segment]).normalized;
            Vector2 normal = new Vector2(-direction.y, direction.x);
            float angle = rng.Range(new Vector2(-Mathf.PI, Mathf.PI));

            blotches[i] = new Blotch
            {
                Center = Vector2.Lerp(points[segment], points[segment + 1], at - segment)
                         + normal * rng.Range(new Vector2(-0.5f, 0.5f) * lineWidth),
                Axis = new Vector2(
                    direction.x * Mathf.Cos(angle) - direction.y * Mathf.Sin(angle),
                    direction.x * Mathf.Sin(angle) + direction.y * Mathf.Cos(angle)),
                Half = 0.5f * new Vector2(rng.Range(BlotchSizeRange), rng.Range(BlotchSizeRange)),
                Color = Color.HSVToRGB(
                    rng.Value, rng.Range(BlotchSaturationRange), rng.Range(BlotchValueRange))
            };
        }
    }

    float Curvature(ArenaRandom rng)
    {
        return rng.Range(new Vector2(-maxCurvature, maxCurvature));
    }

    void Rasterize()
    {
        Array.Clear(coverage, 0, coverage.Length);
        for (int i = 1; i < count; i++) Stroke(points[i - 1], points[i]);

        // RGB24: the texture has no alpha channel at all, so it is always opaque.
        Color32 background = backgroundColor;
        Color32 line = lineColor;
        for (int i = 0; i < pixels.Length; i++)
            pixels[i] = Color32.Lerp(background, line, coverage[i] / 255f);

        for (int i = 0; i < blotchCount; i++) Splat(blotches[i]);

        texture.SetPixels32(pixels);
        texture.Apply(false);
    }

    // One segment of the centerline, antialiased over a texel so the sensor sees a
    // soft edge rather than a staircase.
    void Stroke(Vector2 a, Vector2 b)
    {
        float radius = 0.5f * lineWidth;
        float reach = radius + texel.x;

        int x0 = Pixel(Mathf.Min(a.x, b.x) - reach, size.x, texel.x);
        int x1 = Pixel(Mathf.Max(a.x, b.x) + reach, size.x, texel.x);
        int y0 = Pixel(Mathf.Min(a.y, b.y) - reach, size.y, texel.y);
        int y1 = Pixel(Mathf.Max(a.y, b.y) + reach, size.y, texel.y);

        for (int y = y0; y <= y1; y++)
            for (int x = x0; x <= x1; x++)
            {
                Vector2 c = new Vector2(
                    (x + 0.5f) * texel.x - 0.5f * size.x,
                    (y + 0.5f) * texel.y - 0.5f * size.y);
                float value = Mathf.Clamp01(
                    (radius + 0.5f * texel.x - DistanceToSegment(c, a, b)) / texel.x);

                int index = y * resolution + x;
                byte written = (byte)Mathf.RoundToInt(255f * value);
                if (written > coverage[index]) coverage[index] = written;
            }
    }

    // One blotch straight over the composited line, same soft edge as the stroke.
    void Splat(Blotch blotch)
    {
        Vector2 across = new Vector2(-blotch.Axis.y, blotch.Axis.x);
        float reach = blotch.Half.magnitude + texel.x;

        int x0 = Pixel(blotch.Center.x - reach, size.x, texel.x);
        int x1 = Pixel(blotch.Center.x + reach, size.x, texel.x);
        int y0 = Pixel(blotch.Center.y - reach, size.y, texel.y);
        int y1 = Pixel(blotch.Center.y + reach, size.y, texel.y);

        for (int y = y0; y <= y1; y++)
            for (int x = x0; x <= x1; x++)
            {
                Vector2 d = new Vector2(
                    (x + 0.5f) * texel.x - 0.5f * size.x - blotch.Center.x,
                    (y + 0.5f) * texel.y - 0.5f * size.y - blotch.Center.y);

                float u = Mathf.Abs(Vector2.Dot(d, blotch.Axis)) - blotch.Half.x;
                float v = Mathf.Abs(Vector2.Dot(d, across)) - blotch.Half.y;
                float distance = new Vector2(Mathf.Max(u, 0f), Mathf.Max(v, 0f)).magnitude
                                 + Mathf.Min(Mathf.Max(u, v), 0f);

                float value = Mathf.Clamp01((0.5f * texel.x - distance) / texel.x);
                int index = y * resolution + x;
                pixels[index] = Color32.Lerp(pixels[index], blotch.Color, value);
            }
    }

    int Pixel(float meters, float span, float texelSize)
    {
        return Mathf.Clamp(Mathf.FloorToInt((meters + 0.5f * span) / texelSize), 0, resolution - 1);
    }

    static float DistanceToSegment(Vector2 p, Vector2 a, Vector2 b)
    {
        Vector2 ab = b - a;
        float t = Mathf.Clamp01(Vector2.Dot(p - a, ab) / ab.sqrMagnitude);
        return (p - (a + ab * t)).magnitude;
    }

    public TrackSample Sample(Vector3 world)
    {
        Vector2 p = ToTrack(world);

        float best = float.MaxValue;
        int index = 1;
        float at = 0f;
        for (int i = 1; i < count; i++)
        {
            Vector2 ab = points[i] - points[i - 1];
            float t = Mathf.Clamp01(Vector2.Dot(p - points[i - 1], ab) / ab.sqrMagnitude);
            float distance = (p - (points[i - 1] + ab * t)).sqrMagnitude;
            if (distance < best)
            {
                best = distance;
                index = i;
                at = t;
            }
        }

        Vector2 segment = points[index] - points[index - 1];
        Vector2 nearest = points[index - 1] + segment * at;
        Vector2 direction = segment.normalized;
        Vector2 delta = p - nearest;

        return new TrackSample
        {
            Position = ToWorld(nearest),
            Direction = ToWorldDirection(direction),
            Offset = direction.x * delta.y - direction.y * delta.x,
            Arc = (index - 1 + at) * Step
        };
    }

    // Distance in the pad plane, so how high the robot rides does not count.
    public float DistanceToFinish(Vector3 world)
    {
        return (ToTrack(world) - points[count - 1]).magnitude;
    }

    Vector3 ToWorld(Vector2 track)
    {
        return transform.TransformPoint(new Vector3(track.x / scale.x, 0f, track.y / scale.z));
    }

    Vector3 ToWorldDirection(Vector2 track)
    {
        return transform.TransformDirection(new Vector3(track.x, 0f, track.y));
    }

    Vector2 ToTrack(Vector3 world)
    {
        Vector3 local = transform.InverseTransformPoint(world);
        return new Vector2(local.x * scale.x, local.z * scale.z);
    }

    // Outer rectangle is the printed area, inner one is where the line may run.
    void OnDrawGizmosSelected()
    {
        Gizmos.color = Color.cyan;
        DrawRect(size);
        Gizmos.color = new Color(0f, 1f, 1f, 0.35f);
        DrawRect(size - 2f * margin * Vector2.one);
    }

    void DrawRect(Vector2 extent)
    {
        Vector3 s = transform.lossyScale;
        Vector3 x = 0.5f * extent.x / s.x * Vector3.right;
        Vector3 z = 0.5f * extent.y / s.z * Vector3.forward;

        Vector3 a = transform.TransformPoint(-x - z);
        Vector3 b = transform.TransformPoint(x - z);
        Vector3 c = transform.TransformPoint(x + z);
        Vector3 d = transform.TransformPoint(-x + z);

        Gizmos.DrawLine(a, b);
        Gizmos.DrawLine(b, c);
        Gizmos.DrawLine(c, d);
        Gizmos.DrawLine(d, a);
    }
}
