# GitHub Copilot Instructions for this Project

- **Primary Language:** All generated docstrings, comments, and explanations must be in English.
- **Type Hints:** All Python functions and methods must include explicit type hints for all arguments and return values. Follow PEP 484.
- **Preserve Comments:** Do not modify or remove my existing comments unless specifically asked to. Refactor the code around the comments.
- **Docstrings:** Generate Google-style docstrings for all functions, including `Args:` and `Returns:` sections.
- **Preserve the top menu bar** with its "File" and "Help" entries.
- **UI Definition:** The main window layout is defined in `app/resources/main_window.ui` and loaded at runtime. Do not rebuild the UI manually in Python code.
- **Entry Point:** The application is launched from `run.py` in the project root.
- **File Structure:** The main application logic resides in `app/WhisterTyper.py`. Resource files (icons, sounds, UI) are located in subdirectories within the `app` folder.
- **Preserve `run_app`:** The `run_app` function in `WhisterTyper.py` must be preserved exactly as is.
