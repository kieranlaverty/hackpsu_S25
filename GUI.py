import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw, ImageGrab
import os
import random
import genFace
import cv2
import numpy as np
import pytensor.tensor as pt
from torch import nn
import torch
import torchvision.transforms.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()

        # Encoder (Downsampling)
        self.encoder = nn.Sequential(
            self.conv_block(in_channels, 64),
            self.conv_block(64, 128),
            self.conv_block(128, 256),
            self.conv_block(256, 512),
            self.conv_block(512, 1024)
        )

        # Decoder (Upsampling)
        self.decoder = nn.Sequential(
            self.upconv_block(1024, 512),
            self.upconv_block(512, 256),
            self.upconv_block(256, 128),
            self.upconv_block(128, 64),
            nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder[0](x)
        enc2 = self.encoder[1](enc1)
        enc3 = self.encoder[2](enc2)
        enc4 = self.encoder[3](enc3)
        enc5 = self.encoder[4](enc4)

        # Decoder
        dec1 = self.decoder[0](enc5)
        dec2 = self.decoder[1](dec1 + enc4)
        dec3 = self.decoder[2](dec2 + enc3)
        dec4 = self.decoder[3](dec3 + enc2)
        out = self.decoder[4](dec4 + enc1)

        return out



class ImageSelectorApp:
    def __init__(self, root):
        self.AiPath = None
        self.root = root
        self.root.title("Image Selector with Black Drawing and Boxes")
        self.root.geometry("800x700")
        self.root.configure(bg="#f0f0f0")
        
        self.image_path = ""
        self.original_image = None
        self.display_image = None
        self.drawing_image = None  # For storing the drawing overlay
        
        # Drawing variables
        self.is_drawing = False
        self.last_x = 0
        self.last_y = 0
        self.brush_size = 20
        
        # Box selection variables
        self.is_placing_box = False
        self.start_x = 0
        self.start_y = 0
        self.current_box_id = None
        self.box_width = 100
        self.box_height = 100
        
        self.final_image = None  # The image with drawing applied
        
        # Create main frame
        self.main_frame = tk.Frame(root, bg="#f0f0f0")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create title label
        self.title_label = tk.Label(
            self.main_frame, 
            text="Image Selector with Black Drawing and Boxes", 
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
        self.browse_button.grid(row=0, column=0, padx=5)
        
        # Create drawing mode button
        self.draw_button = tk.Button(
            self.button_frame,
            text="Draw Mode",
            command=self.toggle_draw_mode,
            font=("Arial", 12),
            bg="#fbbc05",
            fg="white",
            padx=10,
            pady=5,
            state=tk.DISABLED
        )
        self.draw_button.grid(row=0, column=1, padx=5)
        
        # Create place box button
        self.place_box_button = tk.Button(
            self.button_frame,
            text="Place Box Mode",
            command=self.toggle_place_box_mode,
            font=("Arial", 12),
            bg="#34a853",
            fg="white",
            padx=10,
            pady=5,
            state=tk.DISABLED
        )
        self.place_box_button.grid(row=0, column=2, padx=5)
        
        # Create random box button
        self.random_box_button = tk.Button(
            self.button_frame,
            text="Random Box",
            command=self.place_random_box,
            font=("Arial", 12),
            bg="#ea4335",
            fg="white",
            padx=10,
            pady=5,
            state=tk.DISABLED
        )
        self.random_box_button.grid(row=0, column=3, padx=5)
        
        # Create clear button
        self.clear_button = tk.Button(
            self.button_frame,
            text="Clear All",
            command=self.clear_all,
            font=("Arial", 12),
            bg="#4285f4",
            fg="white",
            padx=10,
            pady=5,
            state=tk.DISABLED
        )
        self.clear_button.grid(row=0, column=4, padx=5)
        
        # Create size control frame
        self.size_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.size_frame.pack(pady=10)
        
        # Create brush size label
        self.brush_label = tk.Label(
            self.size_frame,
            text="Brush Size:",
            font=("Arial", 12),
            bg="#f0f0f0"
        )
        self.brush_label.grid(row=0, column=0, padx=5)
        
        # Create brush size slider
        self.brush_slider = tk.Scale(
            self.size_frame,
            from_=5,
            to=50,
            orient=tk.HORIZONTAL,
            length=150,
            command=self.update_brush_size,
            bg="#f0f0f0"
        )
        self.brush_slider.set(20)
        self.brush_slider.grid(row=0, column=1, padx=5)
        
        # Create box width label and entry
        self.width_label = tk.Label(
            self.size_frame,
            text="Box Width:",
            font=("Arial", 12),
            bg="#f0f0f0"
        )
        self.width_label.grid(row=0, column=2, padx=5)
        
        self.width_var = tk.StringVar(value="100")
        self.width_entry = tk.Entry(
            self.size_frame,
            textvariable=self.width_var,
            font=("Arial", 12),
            width=5
        )
        self.width_entry.grid(row=0, column=3, padx=5)
        
        # Create box height label and entry
        self.height_label = tk.Label(
            self.size_frame,
            text="Box Height:",
            font=("Arial", 12),
            bg="#f0f0f0"
        )
        self.height_label.grid(row=0, column=4, padx=5)
        
        self.height_var = tk.StringVar(value="100")
        self.height_entry = tk.Entry(
            self.size_frame,
            textvariable=self.height_var,
            font=("Arial", 12),
            width=5
        )
        self.height_entry.grid(row=0, column=5, padx=5)
        
        # Create update size button
        self.update_size_button = tk.Button(
            self.size_frame,
            text="Update Size",
            command=self.update_box_size,
            font=("Arial", 12),
            bg="#4285f4",
            fg="white",
            padx=5,
            pady=2
        )
        self.update_size_button.grid(row=0, column=6, padx=5)
        
        # Create path display frame
        self.path_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.path_frame.pack(fill=tk.X, pady=10)
        
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
        
        # Create canvas frame
        self.canvas_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create canvas for image editing
        self.canvas = tk.Canvas(
            self.canvas_frame,
            bg="#ffffff",
            width=600,
            height=300,
            highlightthickness=1,
            highlightbackground="#cccccc"
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        
        # Create status label
        self.status_label = tk.Label(
            self.main_frame,
            text="Status: Ready",
            font=("Arial", 10),
            bg="#f0f0f0",
            fg="#666666"
        )
        self.status_label.pack(anchor=tk.W, pady=5)
        
        # Create bottom button frame
        self.bottom_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.bottom_frame.pack(pady=10)
        
        # Create save button
        self.save_button = tk.Button(
            self.bottom_frame,
            text="Save Modified Image",
            command=self.save_image,
            font=("Arial", 12),
            bg="#34a853",
            fg="white",
            state=tk.DISABLED,
            padx=10,
            pady=5
        )
        self.save_button.pack(side=tk.LEFT, padx=5)
        
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
        
        # Initialize mode flags
        self.drawing_mode = False
        self.box_placing_mode = False
    
    def update_box_size(self):
        try:
            self.box_width = int(self.width_var.get())
            self.box_height = int(self.height_var.get())
            self.status_label.config(text=f"Status: Box size updated to {self.box_width}x{self.box_height}")
        except ValueError:
            messagebox.showerror("Error", "Box width and height must be integers")
    
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
            self.load_image()
            self.use_button.config(state=tk.NORMAL)
            self.draw_button.config(state=tk.NORMAL)
            self.place_box_button.config(state=tk.NORMAL)
            self.random_box_button.config(state=tk.NORMAL)
            self.status_label.config(text="Status: Image loaded")
    
    def load_image(self):
        try:
            # Load image with PIL
            self.original_image = Image.open(self.image_path)
            
            # Resize image for display (maintaining aspect ratio)
            self.display_image = self.resize_image_for_display(self.original_image)
            
            # Create a transparent drawing layer
            self.drawing_image = Image.new("RGBA", self.display_image.size, (0, 0, 0, 0))
            
            # Display the image on canvas
            self.update_canvas()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def resize_image_for_display(self, image):
        # Resize image for display while maintaining aspect ratio
        width, height = image.size
        canvas_width = self.canvas.winfo_width() or 600
        canvas_height = self.canvas.winfo_height() or 300
        
        # Calculate the scaling factor
        width_ratio = canvas_width / width
        height_ratio = canvas_height / height
        ratio = min(width_ratio, height_ratio)
        
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        # Convert to RGBA if it's not already
        if image.mode != "RGBA":
            image = image.convert("RGBA")
            
        return image.resize((new_width, new_height), Image.LANCZOS)
    
    def update_canvas(self):
        if self.display_image:
            # If we have drawing/boxes, combine with the display image
            if self.final_image:
                # Convert image for display
                photo = ImageTk.PhotoImage(self.final_image)
            else:
                # Display original image
                photo = ImageTk.PhotoImage(self.display_image)
            
            # Clear canvas and display new image
            self.canvas.delete("all")
            self.canvas.config(width=photo.width(), height=photo.height())
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo, tags="image")
            self.canvas.image = photo  # Keep a reference!
    
    def toggle_draw_mode(self):
        self.drawing_mode = not self.drawing_mode
        self.box_placing_mode = False
        
        if self.drawing_mode:
            self.draw_button.config(bg="#ea4335")
            self.place_box_button.config(bg="#34a853")
            self.status_label.config(text="Status: Drawing mode - Draw in black on the image")
            self.clear_button.config(state=tk.NORMAL)
        else:
            self.draw_button.config(bg="#fbbc05")
            self.status_label.config(text="Status: Ready")
    
    def toggle_place_box_mode(self):
        self.box_placing_mode = not self.box_placing_mode
        self.drawing_mode = False
        
        if self.box_placing_mode:
            self.place_box_button.config(bg="#ea4335")
            self.draw_button.config(bg="#fbbc05")
            self.status_label.config(text="Status: Box placement mode - Click and drag to place a box")
            self.clear_button.config(state=tk.NORMAL)
        else:
            self.place_box_button.config(bg="#34a853")
            self.status_label.config(text="Status: Ready")
    
    def update_brush_size(self, value):
        self.brush_size = int(value)
    
    def on_mouse_down(self, event):
        if self.drawing_mode:
            self.is_drawing = True
            self.last_x = event.x
            self.last_y = event.y
            
            # Draw a single dot at the starting position
            self.draw_on_image(event.x, event.y, event.x, event.y)
            
        elif self.box_placing_mode:
            self.is_placing_box = True
            self.start_x = event.x
            self.start_y = event.y
            
            # Create a preview box
            self.current_box_id = self.canvas.create_rectangle(
                self.start_x, self.start_y, 
                self.start_x + 1, self.start_y + 1,
                outline="red", width=2
            )
    
    def on_mouse_drag(self, event):
        if self.drawing_mode and self.is_drawing:
            # Draw line from last position to current position
            self.draw_on_image(self.last_x, self.last_y, event.x, event.y)
            
            # Update last position
            self.last_x = event.x
            self.last_y = event.y
            
        elif self.box_placing_mode and self.is_placing_box and self.current_box_id is not None:
            # Update the preview box
            self.canvas.coords(
                self.current_box_id,
                self.start_x, self.start_y,
                event.x, event.y
            )
    
    def on_mouse_up(self, event):
        if self.drawing_mode:
            self.is_drawing = False
            
        elif self.box_placing_mode and self.is_placing_box and self.current_box_id is not None:
            # Get the box coordinates
            box_coords = self.canvas.coords(self.current_box_id)
            if len(box_coords) == 4:  # Make sure we have valid coordinates
                x1, y1, x2, y2 = box_coords
                
                # Ensure x1,y1 is top-left and x2,y2 is bottom-right
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                
                # Draw the black box on the drawing layer
                self.draw_box_on_image(x1, y1, x2, y2)
            
            # Remove the preview box
            self.canvas.delete(self.current_box_id)
            self.current_box_id = None
            self.is_placing_box = False
    
    def draw_on_image(self, x1, y1, x2, y2):
        if not self.display_image or not self.drawing_image:
            return
            
        # Create a temporary drawing surface
        draw = ImageDraw.Draw(self.drawing_image)
        
        # Draw black line on the transparent drawing layer
        draw.line([(x1, y1), (x2, y2)], fill=(0, 0, 0, 255), width=self.brush_size)
        
        # Combine the original image with the drawing
        self.apply_drawing()
    
    def draw_box_on_image(self, x1, y1, x2, y2):
        if not self.display_image or not self.drawing_image:
            return
            
        # Create a temporary drawing surface
        draw = ImageDraw.Draw(self.drawing_image)
        
        # Draw filled black rectangle on the transparent drawing layer
        draw.rectangle([(x1, y1), (x2, y2)], fill=(0, 0, 0, 255))
        
        # Combine the original image with the drawing
        self.apply_drawing()
        
        # Update status
        width = x2 - x1
        height = y2 - y1
        self.status_label.config(text=f"Status: Box added at ({x1}, {y1}) with size {width}x{height}")
    
    def place_random_box(self):
        if not self.display_image:
            return
        
        # Get display image dimensions
        display_width, display_height = self.display_image.size
        
        # Make sure box size is not larger than image
        box_width = min(self.box_width, display_width)
        box_height = min(self.box_height, display_height)
        
        # Generate random position for box (ensuring it's within image boundaries)
        x = random.randint(0, display_width - box_width)
        y = random.randint(0, display_height - box_height)
        
        # Draw the box
        self.draw_box_on_image(x, y, x + box_width, y + box_height)
        
        # Update status
        self.status_label.config(text=f"Status: Random box placed at ({x}, {y}) with size {box_width}x{box_height}")
    
    def apply_drawing(self):
        if not self.display_image or not self.drawing_image:
            return
            
        # Combine the original display image with the drawing layer
        # Convert display image to RGBA if needed
        if self.display_image.mode != "RGBA":
            base_image = self.display_image.convert("RGBA")
        else:
            base_image = self.display_image.copy()
            
        # Composite the drawing over the image
        self.final_image = Image.alpha_composite(base_image, self.drawing_image)
        
        # Update the canvas
        self.update_canvas()
        
        # Enable the save button
        self.save_button.config(state=tk.NORMAL)
    
    def clear_all(self):
        if not self.display_image:
            return
            
        # Reset drawing layer to transparent
        self.drawing_image = Image.new("RGBA", self.display_image.size, (0, 0, 0, 0))
        
        # Clear the final image
        self.final_image = None
        
        # Update the canvas
        self.update_canvas()
        
        # Update status
        self.status_label.config(text="Status: All drawings and boxes cleared")
        
        # Disable the save button
        self.save_button.config(state=tk.DISABLED)
        self.clear_button.config(state=tk.DISABLED)
    
    def save_image(self):
        if not self.final_image:
            messagebox.showerror("Error", "No modifications to save")
            return
            
        # Ask for save location
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if save_path:
            try:
                # Create full-sized version of the drawing
                # Scale drawing to match original image size
                orig_width, orig_height = self.original_image.size
                display_width, display_height = self.display_image.size
                
                # Convert original image to RGBA
                orig_rgba = self.original_image.convert("RGBA")
                
                
                # Scale up drawing to match original image size
                scaled_drawing = self.drawing_image.resize(
                    (orig_width, orig_height), 
                    Image.LANCZOS
                )
                
                # Combine original image with scaled drawing
                final_full_size = Image.alpha_composite(orig_rgba, scaled_drawing)
                
                # Save the image
                final_full_size.save(save_path)
                self.AiPath = save_path
                messagebox.showinfo("Success", f"Modified image saved to: {save_path}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")
    
    def use_image(self):
        if self.AiPath == None:
            messagebox.showerror("Error", "No image selected for AI processing")
            return
        else:
            try:
                # Call the AI processing function from genFace
                
                image = Image.open(self.AiPath)
                #image = cv2.imread(self.AiPath)
                image_tensor = F.to_tensor(image)
                image_tensor = image_tensor[:3, :, :]

                img = genFace.predict(image_tensor, self.AiPath)
                
                

                
                img = Image.open("D:/hackpsuS25/hackpsu_S25/face/reconstuted.png")
                
                
                
                messagebox.showinfo("Success", "Image sent for AI processing")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process image: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSelectorApp(root)
    root.mainloop()