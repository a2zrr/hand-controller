# C# client (.NET 6+)

`HandInput.cs` receives hand-controller states over UDP. It has no dependency on any engine or
framework, so it works in console apps, WinForms/WPF, ASP.NET, or a game written in any .NET engine.

## Files

- `HandInput.cs`: copy this one file into your project.

## Usage: console app

Create a console app (`dotnet new console`), copy `HandInput.cs` next to `Program.cs`, and replace
`Program.cs` with:

```csharp
using System;
using System.Threading;
using HandInput;

class Program
{
    static void Main()
    {
        // Start listening on UDP port 5005. Create it once, and dispose it when you are done.
        using (var input = new HandInputClient(5005))
        {
            Console.WriteLine("Point your index finger. Press Esc to quit.");
            string last = "";

            while (true)
            {
                // Read the latest state. It is STOP if no message arrived for 300 ms.
                string state = input.CurrentState.ToString();
                if (state != last)
                {
                    Console.WriteLine(state);
                    last = state;
                }

                // Leave the loop when Esc is pressed.
                if (Console.KeyAvailable && Console.ReadKey(true).Key == ConsoleKey.Escape)
                    break;

                Thread.Sleep(50);
            }
        }
    }
}
```

Run `dotnet run` in one terminal, and start hand-controller (or `tools/fake_sender.py`) in another.

## Usage: any game loop

Create the `HandInputClient` once when your game starts, read `CurrentState` in your update
function every frame, and call `Dispose()` when the game exits. Map the state to a direction
however your engine expects (for example, `Up` is negative Y when Y points down on screen).

## Notes

- `CurrentState` never blocks. The receive thread runs in the background.
- Only one program can listen on a port. If the port is in use, the constructor throws.
- The state is `Stop` before the first message and whenever messages stop for 300 ms.
