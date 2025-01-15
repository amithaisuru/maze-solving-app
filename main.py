import tkinter as tk
from collections import deque
from tkinter import ttk

from maze_app import MazeApp

if __name__ == "__main__":
    root = tk.Tk()
    
    # Set minimum window size
    root.minsize(600, 400)
    
    # Configure style
    style = ttk.Style()
    style.configure('TButton', padding=5)
    style.configure('TFrame', padding=5)
    
    app = MazeApp(root)
    
    # Add menu bar
    menubar = tk.Menu(root)
    file_menu = tk.Menu(menubar, tearoff=0)
    file_menu.add_command(label="Reset", command=app.reset_maze)
    file_menu.add_command(label="Stop Animation", command=app.stop_animation)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.quit)
    menubar.add_cascade(label="File", menu=file_menu)
    root.config(menu=menubar)
    
    root.mainloop()