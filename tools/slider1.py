import tkinter as tk

def update_value1(val):
    print(f"Slider 1 value: {val}")

def update_value2(val):
    print(f"Slider 2 value: {val}")

def update_value3(val):
    print(f"Slider 3 value: {val}")

root = tk.Tk()
root.title("Multiple Sliders Example")
root.geometry("600x300")  # Set the width and height of the window

# Slider 1
label1 = tk.Label(root, text="Slider 1")
label1.pack(fill=tk.X, padx=10, pady=5)
slider1 = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL, command=update_value1)
slider1.pack(fill=tk.X, padx=10, pady=5)

# Slider 2
label2 = tk.Label(root, text="Slider 2")
label2.pack(fill=tk.X, padx=10, pady=5)
slider2 = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL, command=update_value2)
slider2.pack(fill=tk.X, padx=10, pady=5)

# Slider 3
label3 = tk.Label(root, text="Slider 3")
label3.pack(fill=tk.X, padx=10, pady=5)
slider3 = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL, command=update_value3)
slider3.pack(fill=tk.X, padx=10, pady=5)

root.mainloop()