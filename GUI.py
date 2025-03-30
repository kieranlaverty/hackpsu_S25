import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import os

class ImageSelectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Selector")
        self.root.geometry("600x400")
        self.root.configure(bg="#f0f0f0")
        
        self.image_path = ""
        
        # Create main frame
        self.main_frame = tk.Frame(root, bg="#f0f0f0")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create title label
        self.title_label = tk.Label(
            self.main_frame, 
            text="Image Selector", 
            font=("Arial", 18, "bold"),
            bg="#f0f0f0"
        )
        self.title_label.pack(pady=(0, 20))
        
        # Create button frame
        self.button_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.button_frame.pack(pady=10)
        
        # Create browse button
        self.browse_button = tk.Button(
            self.button_frame,
            text="Browse for Image",
            command=self.browse_image,
            font=("Arial", 12),
            bg="#4285f4",
            fg="white",
            padx=10,
            pady=5
        )
        self.browse_button.pack()
        
        # Create path display frame
        self.path_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.path_frame.pack(fill=tk.X, pady=20)
        
        # Create path label
        self.path_label = tk.Label(
            self.path_frame,
            text="Selected Image Path:",
            font=("Arial", 12),
            bg="#f0f0f0"
        )
        self.path_label.pack(anchor=tk.W)
        
        # Create path entry
        self.path_var = tk.StringVar()
        self.path_entry = tk.Entry(
            self.path_frame,
            textvariable=self.path_var,
            font=("Arial", 10),
            width=50
        )
        self.path_entry.pack(fill=tk.X, pady=(5, 0))
        
        # Create preview frame
        self.preview_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.preview_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create preview label
        self.preview_label = tk.Label(
            self.preview_frame,
            text="Image Preview",
            font=("Arial", 12),
            bg="#f0f0f0"
        )
        self.preview_label.pack()
        
        # Create image preview label
        self.image_label = tk.Label(self.preview_frame, bg="#f0f0f0")
        self.image_label.pack(pady=10)
        
        # Create bottom button frame
        self.bottom_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.bottom_frame.pack(pady=10)
        
        # Create use image button
        self.use_button = tk.Button(
            self.bottom_frame,
            text="Use this Image",
            command=self.use_image,
            font=("Arial", 12),
            bg="#34a853",
            fg="white",
            state=tk.DISABLED,
            padx=10,
            pady=5
        )
        self.use_button.pack(side=tk.LEFT, padx=5)
        
        # Create cancel button
        self.cancel_button = tk.Button(
            self.bottom_frame,
            text="Cancel",
            command=self.root.quit,
            font=("Arial", 12),
            bg="#ea4335",
            fg="white",
            padx=10,
            pady=5
        )
        self.cancel_button.pack(side=tk.LEFT, padx=5)
    
    def browse_image(self):
        # Open file dialog
        filetypes = [
            ("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp"),
            ("All files", "*.*")
        ]
        
        selected_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=filetypes
        )
        
        if selected_path:
            self.image_path = selected_path
            self.path_var.set(selected_path)
            self.load_preview()
            self.use_button.config(state=tk.NORMAL)
        
    def load_preview(self):
        try:
            # Load image with PIL
            image = Image.open(self.image_path)
            
            # Resize image for preview (maintaining aspect ratio)
            width, height = image.size
            max_size = 200
            
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
                
            image = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Update image label
            self.image_label.config(image=photo)
            self.image_label.image = photo  # Keep a reference!
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def use_image(self):
        # Here you would add code to process the selected image
        messagebox.showinfo("Success", f"Selected image: {self.image_path}")
        # You could access self.image_path in your application logic

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSelectorApp(root)
    root.mainloop()