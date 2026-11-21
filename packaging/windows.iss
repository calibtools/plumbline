; Inno Setup script. Build with:  iscc /DAppVersion=0.1.0 packaging\windows.iss
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{6B0E5F0A-3C0B-4A7E-9C55-0C6D4B6C7A11}
AppName=Plumbline
AppVersion={#AppVersion}
AppPublisher=Calib.Tools
AppPublisherURL=https://calib.tools
AppSupportURL=https://github.com/calibtools/plumbline/issues
DefaultDirName={autopf}\Plumbline
DefaultGroupName=Plumbline
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist\installer
OutputBaseFilename=Plumbline-{#AppVersion}-windows-x64-setup
LicenseFile=..\LICENSE
; The whole point of this file: one solid LZMA2 stream over all files.
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=4
WizardStyle=modern
UninstallDisplayName=Plumbline by Calib.Tools

[Files]
Source: "..\dist\Plumbline\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\Plumbline"; Filename: "{app}\Plumbline.exe"
Name: "{autodesktop}\Plumbline"; Filename: "{app}\Plumbline.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Run]
Filename: "{app}\Plumbline.exe"; Description: "Start Plumbline"; Flags: nowait postinstall skipifsilent
