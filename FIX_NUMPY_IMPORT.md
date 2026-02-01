# Quick Fix for "Import numpy could not be resolved"

## Solution 1: Select Python Interpreter in VS Code

1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
2. Type: `Python: Select Interpreter`
3. Choose: `C:\Python314\python.exe` (or your conda environment)

## Solution 2: Install numpy in current environment

```bash
pip install numpy
```

## Solution 3: Restart VS Code

After selecting the interpreter, restart VS Code to refresh the language server.

## Solution 4: If using Jupyter Notebook

In the notebook, run:
```python
%pip install numpy
```

Then restart the kernel: `Kernel → Restart Kernel`

## Note

The error is just a VS Code linting issue. Your code will still run if numpy is installed. The `.vscode/settings.json` file has been created to help VS Code find the correct interpreter.

