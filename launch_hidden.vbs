Option Explicit

Dim shell, fso, folder, scriptPath, command, exitCode, comspec
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

folder = fso.GetParentFolderName(WScript.ScriptFullName)
scriptPath = fso.BuildPath(folder, "jp_subtitle_translator.py")
comspec = shell.ExpandEnvironmentStrings("%ComSpec%")

command = Chr(34) & comspec & Chr(34) & " /c pyw -3 " & Chr(34) & scriptPath & Chr(34) & " --gui"
exitCode = shell.Run(command, 0, True)

If exitCode <> 0 Then
  command = Chr(34) & comspec & Chr(34) & " /c pythonw " & Chr(34) & scriptPath & Chr(34) & " --gui"
  exitCode = shell.Run(command, 0, True)
End If

If exitCode <> 0 Then
  MsgBox "Startup failed. Please install Python 3 and make sure pyw or pythonw can run.", vbExclamation, "Subtitle Translator"
End If
