using System;
using UnityEngine;

// Command line arguments use a double dash to avoid colliding with Unity's own.
public static class Args
{
    public static int GetInt(string key, int editorFallback)
    {
        string raw = Find(key);
        if (raw != null) return int.Parse(raw);
        if (Application.isEditor) return editorFallback;
        throw new Exception("missing command line argument --" + key);
    }

    static string Find(string key)
    {
        string[] args = Environment.GetCommandLineArgs();
        for (int i = 0; i + 1 < args.Length; i++)
            if (args[i] == "--" + key) return args[i + 1];
        return null;
    }
}
