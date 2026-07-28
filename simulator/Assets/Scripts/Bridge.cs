using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

// Python drives the simulation: one request advances one control step.
public class Bridge : MonoBehaviour
{
    const int Version = 1;

    [SerializeField] Arena arenaPrefab;
    [SerializeField] float arenaSpacing = 20f;
    [SerializeField] float controlPeriod = 0.05f;
    [SerializeField] float physicsDt = 0.005f;
    [SerializeField] int editorPort = 5005;
    [SerializeField] int editorArenas = 1;

    Arena[] arenas;
    int substeps;
    TcpListener listener;
    TcpClient client;
    NetworkStream stream;
    string buffer = "";
    readonly byte[] chunk = new byte[1 << 16];

    void Awake()
    {
        if (arenaPrefab == null) throw new Exception("Bridge.arenaPrefab is not set");

        Physics.simulationMode = SimulationMode.Script;
        Time.fixedDeltaTime = physicsDt;
        substeps = Mathf.RoundToInt(controlPeriod / physicsDt);

        int count = Args.GetInt("arenas", editorArenas);
        arenas = new Arena[count];
        for (int i = 0; i < count; i++)
            arenas[i] = Instantiate(arenaPrefab, new Vector3(i * arenaSpacing, 0f, 0f), Quaternion.identity);

        int port = Args.GetInt("port", editorPort);
        listener = new TcpListener(IPAddress.Loopback, port);
        listener.Start();
        Debug.Log("bridge listening on port " + port + ", " + count + " arenas");
    }

    void Update()
    {
        if (client == null)
        {
            if (!listener.Pending()) return;
            client = listener.AcceptTcpClient();
            client.NoDelay = true;
            stream = client.GetStream();
            Debug.Log("Connected");
            return;
        }

        if (!stream.DataAvailable) return;

        string line = ReadLine();
        if (line == null)
        {
            Disconnect();
            return;
        }

        Debug.Log(line);
        string response = Handle(JsonUtility.FromJson<Request>(line));
        Debug.Log(response);
        if (response != null) Send(response);
    }

    string Handle(Request request)
    {
        switch (request.cmd)
        {
            case "handshake":
                if (request.version != Version)
                    throw new Exception("protocol version mismatch: python " + request.version + ", unity " + Version);
                // Python owns the seed root; arenas only derive from it when they
                // auto reset, which happens without python in the loop.
                for (int i = 0; i < arenas.Length; i++) arenas[i].Initialize(i, request.session_seed);
                return JsonUtility.ToJson(new HandshakeResponse
                {
                    version = Version,
                    arenas = arenas.Length,
                    dt = controlPeriod,
                    obs_dim = RobotController.ObsDim,
                    action_dim = RobotController.ActionDim
                });

            case "reset":
                if (request.seeds != null && request.seeds.Length != arenas.Length)
                    throw new Exception("expected " + arenas.Length + " seeds, got " + request.seeds.Length);
                for (int i = 0; i < arenas.Length; i++)
                {
                    int seed = request.seeds != null ? request.seeds[i] : arenas[i].NextSeed();
                    arenas[i].ResetEpisode(request.randomize_scenario, request.randomize_physics, seed);
                }
                return JsonUtility.ToJson(Gather());

            case "step":
                StepAll(request.actions);
                return JsonUtility.ToJson(Gather());

            case "quit":
                Disconnect();
                Application.Quit();
                return null;
        }

        throw new Exception("unknown command: " + request.cmd);
    }

    void StepAll(float[] actions)
    {
        int expected = arenas.Length * RobotController.ActionDim;
        if (actions == null || actions.Length != expected)
            throw new Exception("expected " + expected + " action values");

        for (int i = 0; i < arenas.Length; i++)
            arenas[i].ApplyAction(actions[i * 2], actions[i * 2 + 1]);

        for (int s = 0; s < substeps; s++)
        {
            for (int i = 0; i < arenas.Length; i++) arenas[i].Tick(physicsDt);
            Physics.Simulate(physicsDt);
        }

        for (int i = 0; i < arenas.Length; i++) arenas[i].AdvanceStep();
    }

    StateResponse Gather()
    {
        int dim = RobotController.ObsDim;
        StateResponse state = new StateResponse
        {
            obs = new float[arenas.Length * dim],
            terminal_obs = new float[arenas.Length * dim],
            reward = new float[arenas.Length],
            terminated = new bool[arenas.Length],
            truncated = new bool[arenas.Length],
            episode = new int[arenas.Length],
            step = new int[arenas.Length]
        };

        for (int i = 0; i < arenas.Length; i++)
        {
            // obs is the new episode's first observation when the arena auto reset;
            // terminal_obs is the last observation of the episode that just ended,
            // left as zeros where no episode ended so stale values cannot be read.
            arenas[i].Observe(state.obs, i * dim);
            if (arenas[i].Terminated || arenas[i].Truncated)
                arenas[i].ObserveTerminal(state.terminal_obs, i * dim);
            state.reward[i] = arenas[i].Reward;
            state.terminated[i] = arenas[i].Terminated;
            state.truncated[i] = arenas[i].Truncated;
            state.episode[i] = arenas[i].Episode;
            state.step[i] = arenas[i].Step;
        }

        return state;
    }

    string ReadLine()
    {
        while (true)
        {
            int newline = buffer.IndexOf('\n');
            if (newline >= 0)
            {
                string line = buffer.Substring(0, newline);
                buffer = buffer.Substring(newline + 1);
                return line;
            }

            int read = stream.Read(chunk, 0, chunk.Length);
            if (read <= 0) return null;
            buffer += Encoding.UTF8.GetString(chunk, 0, read);
        }
    }

    void Send(string message)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(message + "\n");
        stream.Write(bytes, 0, bytes.Length);
    }

    void Disconnect()
    {
        if (client == null) return;
        client.Close();
        client = null;
        stream = null;
        buffer = "";
    }

    void OnDestroy()
    {
        Disconnect();
        if (listener != null) listener.Stop();
    }
}
