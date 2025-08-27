# 🎹 AI Auto Accompaniment for Digital Piano

This project provides **real-time automatic accompaniment** for digital pianos via MIDI.  
It listens to your playing, analyzes the melody (key, tempo, genre, style), and generates a matching accompaniment on the fly.

Tested on **Medeli SP-A500**, but should work with most digital pianos that support MIDI.

---
INSTALL GUID
For install Downloat .zip with project - klick code<> then download zip
Next unzip it in new directory
Then in directory where you unzip project click open in terminal


## ✨ Features
- **Automatic MIDI port detection** (ignores `Midi Through`).
- **I/O check** before starting:  
  - waits for a key press on input,  
  - sends a test chord on output.
- **15-second analysis** of your playing:  
  - key detection,  
  - tempo estimation,  
  - genre/style recognition.
- **Live accompaniment generation** in sync with your melody.
- **Safe shutdown**: sends *All Notes Off* to prevent hanging notes.
- **Friendly console messages**, e.g.:
