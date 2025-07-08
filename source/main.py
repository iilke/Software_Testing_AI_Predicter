import tkinter as tk
from tkinter import ttk
import re
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tkinter import filedialog
import csv
from fpdf import FPDF
from datetime import datetime
import sys, os

current_sort_mode = 0  # 0: Priority, 1: Duration, 2: Pass Rate, 3: TCID
current_results = []

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


# Load pre-trained ML models and preprocessors
with open(resource_path("clf.pkl"), "rb") as f:
    clf = pickle.load(f)
with open(resource_path("reg.pkl"), "rb") as f:
    reg = pickle.load(f)
with open(resource_path("tfidf.pkl"), "rb") as f:
    tfidf = pickle.load(f)
with open(resource_path("scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(resource_path("X_features.pkl"), "rb") as f:
    X_features = pickle.load(f)
with open(resource_path("test_ids.pkl"), "rb") as f:
    test_ids = pickle.load(f)
with open(resource_path("similar_case_details.pkl"), "rb") as f:
    similar_case_details = pickle.load(f)


# Modern Color Palette - Soft Pastels
COLORS = {
    'bg_primary': '#f8fafc',      # Very light blue-gray
    'bg_secondary': '#f1f5f9',    # Light blue-gray
    'bg_card': '#ffffff',         # Pure white
    'accent_soft': '#e0e7ff',     # Soft lavender -hover
    'accent_light': '#c7d2fe',    # Light indigo -main buttons
    'text_primary': '#1e293b',    # Dark slate
    'text_secondary': '#64748b',  # Medium slate
    'text_muted': '#94a3b8',      # Light slate
    'success': '#dcfce7',         # Soft green
    'warning': '#fef3c7',         # Soft yellow
    'error': '#fee2e2',           # Soft red
    'border': '#e2e8f0',          # Light border
    'shadow': '#f1f5f9',          # Subtle shadow
    'order_page_btns' : '#ffe1cc',#peach
    'order_page_hover': '#fff5ee' #lighter peach
}

class ModernButton(tk.Frame):
    def __init__(self, parent, text, command=None, bg_color=COLORS['accent_light'], 
                 text_color=COLORS['text_primary'], hover_color=COLORS['accent_soft'],
                 font=('Inter', 11), padding=(20, 12), **kwargs):
        super().__init__(parent, bg=parent['bg'], **kwargs)
        
        self.command = command
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.text_color = text_color
        
        # Create button frame with rounded appearance
        self.button_frame = tk.Frame(self, bg=bg_color, relief='flat', bd=0)
        self.button_frame.pack(fill='both', expand=True, padx=2, pady=2)
        
        # Button label
        self.label = tk.Label(
            self.button_frame, 
            text=text, 
            bg=bg_color, 
            fg=text_color,
            font=font,
            cursor='hand2'
        )
        self.label.pack(padx=padding[0], pady=padding[1])
        
        # Bind events
        self.bind_events()
    
    def bind_events(self):
        widgets = [self, self.button_frame, self.label]
        for widget in widgets:
            widget.bind('<Button-1>', self.on_click)
            widget.bind('<Enter>', self.on_enter)
            widget.bind('<Leave>', self.on_leave)
    
    def on_click(self, event):
        if self.command:
            self.command()
    
    def on_enter(self, event):
        self.button_frame.config(bg=self.hover_color)
        self.label.config(bg=self.hover_color)
    
    def on_leave(self, event):
        self.button_frame.config(bg=self.bg_color)
        self.label.config(bg=self.bg_color)

class ModernCard(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=COLORS['bg_card'], relief='flat', bd=0, **kwargs)
        
        # Add subtle shadow effect with multiple frames
        shadow_frame = tk.Frame(parent, bg=COLORS['shadow'], height=2)
        shadow_frame.place(in_=self, x=3, y=3, relwidth=1, relheight=1)
        self.lift()

class ModernEntry(tk.Frame):
    def __init__(self, parent, height=1, **kwargs):
        super().__init__(parent, bg=parent['bg'])
        
        # Create entry with modern styling
        self.text_widget = tk.Text(
            self,
            height=height,
            bg=COLORS['bg_card'],
            fg=COLORS['text_primary'],
            font=('Inter', 11),
            relief='flat',
            bd=0,
            padx=15,
            pady=10,
            wrap='word',
            selectbackground=COLORS['accent_soft'],
            insertbackground=COLORS['text_primary']
        )
        
        # Create border frame
        border_frame = tk.Frame(self, bg=COLORS['border'], height=1)
        border_frame.pack(fill='x', side='bottom')
        
        self.text_widget.pack(fill='both', expand=True, padx=1, pady=(1,0))
        
        # Focus events for modern interaction
        self.text_widget.bind('<FocusIn>', self.on_focus_in)
        self.text_widget.bind('<FocusOut>', self.on_focus_out)
    
    def on_focus_in(self, event):
        self.config(bg=COLORS['accent_soft'])
    
    def on_focus_out(self, event):
        self.config(bg=COLORS['bg_primary'])
    
    def get(self, *args):
        return self.text_widget.get(*args)
    
    def insert(self, *args):
        return self.text_widget.insert(*args)
    
    def delete(self, *args):
        return self.text_widget.delete(*args)

class ExpandableText(tk.Frame):
    def __init__(self, parent, text, max_chars=80, **kwargs):
        super().__init__(parent, **kwargs)
        self.full_text = text
        self.max_chars = max_chars
        self.is_expanded = False
        
        self.text_var = tk.StringVar()
        self.update_display()
        
        # Text label
        self.text_label = tk.Label(
            self,
            textvariable=self.text_var,
            font=('Inter', 9),
            bg=kwargs.get('bg', COLORS['bg_secondary']),
            fg=COLORS['text_muted'],
            wraplength=250,
            justify='left',
            anchor='w'
        )
        self.text_label.pack(anchor='w', fill='x')
        
        # Show more/less button if text is long
        if len(self.full_text) > self.max_chars:
            self.toggle_btn = tk.Label(
                self,
                text="show more",
                font=('Inter', 9, 'underline'),
                bg=kwargs.get('bg', COLORS['bg_secondary']),
                fg=COLORS['accent_light'],
                cursor='hand2'
            )
            self.toggle_btn.pack(anchor='w', pady=(2, 0))
            self.toggle_btn.bind('<Button-1>', self.toggle_text)
    
    def update_display(self):
        if self.is_expanded or len(self.full_text) <= self.max_chars:
            self.text_var.set(self.full_text)
        else:
            self.text_var.set(self.full_text[:self.max_chars] + "...")
    
    def toggle_text(self, event=None):
        self.is_expanded = not self.is_expanded
        self.update_display()
        
        if hasattr(self, 'toggle_btn'):
            self.toggle_btn.config(text="show less" if self.is_expanded else "show more")



# Create main window with modern styling
root = tk.Tk()
root.title("AI Test Case Predictor")
root.geometry("900x700")
root.configure(bg=COLORS['bg_primary'])

# Configure modern fonts
try:
    root.option_add('*Font', 'Inter 10')
except:
    root.option_add('*Font', 'Arial 10')

def create_gradient_frame(parent, color1, color2, height=4):
    """Create a subtle gradient effect using multiple frames"""
    gradient_frame = tk.Frame(parent, height=height, bg=color1)
    for i in range(height):
        line = tk.Frame(gradient_frame, height=1, bg=color1 if i < height//2 else color2)
        line.pack(fill='x')
    return gradient_frame

def show_result_page(test_input, parent=root, on_return=None):

    for widget in parent.winfo_children():
        widget.destroy()
    

    parent.configure(bg=COLORS['bg_primary'])
    
    if on_return is None:
            on_return = show_main_screen

    # ML Prediction Results
    results = parse_and_predict(test_input)
    if not results:
        print("Prediction failed.")
        return
    
    # Main container with padding
    main_container = tk.Frame(parent, bg=COLORS['bg_primary'])
    main_container.pack(fill='both', expand=True, padx=30, pady=20)
    
    # Header section
    header_frame = tk.Frame(main_container, bg=COLORS['bg_primary'])
    header_frame.pack(fill='x', pady=(0, 30))
    
    # Return button - modern style
    return_btn = ModernButton(
        header_frame, 
        text="← Return", 
        command=on_return,
        bg_color=COLORS['accent_soft'],
        hover_color=COLORS['accent_light'],
        font=('Inter', 10),
        padding=(16, 8)
    )
    return_btn.pack(side='left')
    
    # Title with modern typography
    title_label = tk.Label(
        header_frame, 
        text="Prediction Results", 
        font=('Inter', 24, 'normal'), 
        bg=COLORS['bg_primary'],
        fg=COLORS['text_primary']
    )
    title_label.pack(side='left', padx=(30, 0))
    
    # Subtitle
    subtitle_label = tk.Label(
        header_frame, 
        text="AI-powered test case analysis and recommendations", 
        font=('Inter', 12), 
        bg=COLORS['bg_primary'],
        fg=COLORS['text_secondary']
    )
    subtitle_label.pack(side='left', padx=(15, 0), pady=(5, 0))
    
    # Content area with two columns
    content_frame = tk.Frame(main_container, bg=COLORS['bg_primary'])
    content_frame.pack(fill='both', expand=True)
    
    # Left column - Main results
    left_column = ModernCard(content_frame)
    left_column.pack(side='left', fill='both', expand=True, padx=(0, 15))
    
    # Right column - Similar cases
    right_column = ModernCard(content_frame)
    right_column.pack(side='right', fill='y', padx=(15, 0))
    right_column.config(width=320)
    
    # Parse test input
    try:
        parts = test_input.strip().split(",", 2)
        tcid = parts[0].strip()
        steps_raw = parts[2].strip()
        step_lines = [s.strip() for s in steps_raw.split(";") if s.strip()]
    except:
        tcid = "Unknown"
        step_lines = ["Parsing error"]
    
    # Left column content
    left_content = tk.Frame(left_column, bg=COLORS['bg_card'])
    left_content.pack(fill='both', expand=True, padx=25, pady=25)
    
    # Test Case Info Section
    tk.Label(
        left_content, 
        text="Test Case Details", 
        font=('Inter', 16, 'normal'), 
        bg=COLORS['bg_card'],
        fg=COLORS['text_primary']
    ).pack(anchor='w', pady=(0, 15))
    
    # TCID with modern styling
    tcid_frame = tk.Frame(left_content, bg=COLORS['accent_soft'])
    tcid_frame.pack(fill='x', pady=(0, 20))
    tk.Label(
        tcid_frame, 
        text=f"Test Case ID: {tcid}", 
        font=('Inter', 12, 'normal'), 
        bg=COLORS['accent_soft'],
        fg=COLORS['text_primary']
    ).pack(padx=15, pady=8, anchor='w')
    
    # Steps section
    tk.Label(
        left_content, 
        text="Test Steps", 
        font=('Inter', 14, 'normal'), 
        bg=COLORS['bg_card'],
        fg=COLORS['text_primary']
    ).pack(anchor='w', pady=(0, 10))
    
    # Steps container with scrollable text widget
    steps_frame = tk.Frame(left_content, bg=COLORS['bg_secondary'], height=200)
    steps_frame.pack(fill='x', pady=(0, 25))
    steps_frame.pack_propagate(False)
    
    # Create scrollable text widget for steps
    steps_text = tk.Text(
        steps_frame,
        bg=COLORS['bg_secondary'],
        fg=COLORS['text_secondary'],
        font=('Inter', 11),
        relief='flat',
        bd=0,
        padx=15,
        pady=10,
        wrap='word',
        height=10,
        state='disabled',
        cursor='arrow'
    )
    
    # Scrollbar for steps
    steps_scrollbar = tk.Scrollbar(steps_frame, orient='vertical', command=steps_text.yview)
    steps_text.config(yscrollcommand=steps_scrollbar.set)
    
    # Pack scrollbar and text
    steps_scrollbar.pack(side='right', fill='y')
    steps_text.pack(side='left', fill='both', expand=True)
    
    # Insert all steps
    steps_text.config(state='normal')
    for idx, step in enumerate(step_lines):
        steps_text.insert('end', f"• {step}\n\n")
    steps_text.config(state='disabled')
    
    # Predictions section with cards
    tk.Label(
        left_content, 
        text="AI Predictions", 
        font=('Inter', 16, 'normal'), 
        bg=COLORS['bg_card'],
        fg=COLORS['text_primary']
    ).pack(anchor='w', pady=(0, 15))
    
    # Predictions container
    pred_container = tk.Frame(left_content, bg=COLORS['bg_card'])
    pred_container.pack(fill='x')
    
    # Duration prediction card
    duration_card = tk.Frame(pred_container, bg=COLORS['success'])
    duration_card.pack(side='left', fill='x', expand=True, padx=(0, 10))
    
    tk.Label(
        duration_card,
        text="Estimated Duration",
        font=('Inter', 11, 'normal'),
        bg=COLORS['success'],
        fg=COLORS['text_secondary']
    ).pack(pady=(15, 5))
    
    tk.Label(
        duration_card,
        text=f"{results['predicted_duration']}s",
        font=('Inter', 20, 'normal'),
        bg=COLORS['success'],
        fg=COLORS['text_primary']
    ).pack(pady=(0, 15))
    
    # Pass rate prediction card
    passrate_card = tk.Frame(pred_container, bg=COLORS['warning'])
    passrate_card.pack(side='right', fill='x', expand=True, padx=(10, 0))
    
    tk.Label(
        passrate_card,
        text="Estimated Pass Rate",
        font=('Inter', 11, 'normal'),
        bg=COLORS['warning'],
        fg=COLORS['text_secondary']
    ).pack(pady=(15, 5))
    
    tk.Label(
        passrate_card,
        text=f"{results['predicted_passrate']}%",
        font=('Inter', 20, 'normal'),
        bg=COLORS['warning'],
        fg=COLORS['text_primary']
    ).pack(pady=(0, 15))
    
    # Right column - Similar test cases
    right_content = tk.Frame(right_column, bg=COLORS['bg_card'])
    right_content.pack(fill='both', expand=True, padx=20, pady=20)
    
    # Similar cases header
    tk.Label(
        right_content,
        text="Similar Test Cases",
        font=('Inter', 14, 'normal'),
        bg=COLORS['bg_card'],
        fg=COLORS['text_primary']
    ).pack(anchor='w', pady=(0, 15))
    
    # Similar cases list
    for i, case_id in enumerate(results["similar_cases"]):
        detail = similar_case_details.get(case_id, {})
        steps = detail.get("steps", "N/A")
        duration = detail.get("duration", "N/A")
        result = detail.get("result", "N/A")
        
        # Case card
        case_card = tk.Frame(right_content, bg=COLORS['bg_secondary'])
        case_card.pack(fill='x', pady=(0, 12))
        
        case_content = tk.Frame(case_card, bg=COLORS['bg_secondary'])
        case_content.pack(fill='x', padx=12, pady=12)
        
        # Case ID
        tk.Label(
            case_content,
            text=case_id,
            font=('Inter', 11, 'normal'),
            bg=COLORS['bg_secondary'],
            fg=COLORS['text_primary']
        ).pack(anchor='w')
        
        # Case details
        details_text = f"Duration: {duration}s | Result: {result}"
        tk.Label(
            case_content,
            text=details_text,
            font=('Inter', 9),
            bg=COLORS['bg_secondary'],
            fg=COLORS['text_secondary']
        ).pack(anchor='w', pady=(2, 0))
        
        # Steps with expandable text
        steps_expandable = ExpandableText(
            case_content,
            text=steps,
            max_chars=60,
            bg=COLORS['bg_secondary']
        )
        steps_expandable.pack(anchor='w', fill='x', pady=(5, 0))



def validate_detailed_test_case(line: str) -> bool:
    try:
        parts = [p.strip() for p in line.strip().split(",", 2)]
        if len(parts) != 3:
            return False
        tcid, num_steps_str, step_desc = parts
        if not tcid or not num_steps_str.isdigit():
            return False
        num_steps = int(num_steps_str)
        found_steps = re.findall(r'\b\d+-', step_desc)
        if len(found_steps) != num_steps:
            return False
        return True
    except:
        return False



def on_predict():
    test_input = input_text.get("1.0", tk.END).strip()
    if not test_input:
        result_label.config(text="Please enter a test case first.", fg=COLORS['text_secondary'])
        return
    if validate_detailed_test_case(test_input):
        for widget in root.winfo_children():
            widget.destroy()
        show_result_page(test_input)
    else:
        result_label.config(text="Invalid format! Please check your input.", fg='#ef4444')



def on_order_multiple():
    OrderWindow(root)




def parse_and_predict(test_input):
    try:
        parts = test_input.strip().split(",", 2)
        if len(parts) != 3:
            return None
        tcid = parts[0].strip()
        num_steps_str = parts[1].strip()
        steps_raw = parts[2].strip()
        if not tcid or not num_steps_str.isdigit():
            return None
        num_steps = int(num_steps_str)
        found_steps = re.findall(r'\b\d+-', steps_raw)
        if len(found_steps) != num_steps:
            return None
        step_keywords = re.sub(r'\b\d+-', '', steps_raw).strip()
        X_keywords = tfidf.transform([step_keywords]).toarray()
        X_steps = scaler.transform([[num_steps]])
        X_input = np.hstack([X_keywords, X_steps])
        pred_duration = reg.predict(X_input)[0]
        pred_passrate = clf.predict_proba(X_input)[0][1] * 100
        sims = cosine_similarity(X_input, X_features)[0]
        top_indices = sims.argsort()[-4:-1][::-1]
        similar_cases = [test_ids[i] for i in top_indices]
        return {
            "predicted_duration": round(pred_duration, 2),
            "predicted_passrate": round(pred_passrate, 1),
            "similar_cases": similar_cases
        }
    except Exception as e:
        print("Error in parse_and_predict:", e)
        return None



def run_main_screen():
    global input_text, result_label
    
    # Main container with better spacing
    main_container = tk.Frame(root, bg=COLORS['bg_primary'])
    main_container.pack(fill='both', expand=True, padx=40, pady=30)
    
    # Header section
    header_frame = tk.Frame(main_container, bg=COLORS['bg_primary'])
    header_frame.pack(fill='x', pady=(0, 40))
    
    # Order button – RIGHT side
    order_btn = ModernButton(
        header_frame,  # or root, depending on where you want to place it
        text="⚙ Order Test Cases",
        command=show_order_page,
        bg_color=COLORS['order_page_hover'],
        hover_color=COLORS['order_page_btns'],
        font=('Inter', 10),
        padding=(16, 8)
    )
    order_btn.place(relx=1.0, rely=0.1, anchor='ne')  # sağ üst köşeye yakın


    # App title with modern typography
    title_label = tk.Label(
        header_frame,
        text="AI Test Case Predictor",
        font=('Inter', 28, 'normal'),
        bg=COLORS['bg_primary'],
        fg=COLORS['text_primary']
    )
    title_label.pack()
    
    # Subtitle
    subtitle_label = tk.Label(
        header_frame,
        text="Intelligent test case analysis powered by machine learning",
        font=('Inter', 14),
        bg=COLORS['bg_primary'],
        fg=COLORS['text_secondary']
    )
    subtitle_label.pack(pady=(8, 0))
    
    # Input section in a modern card
    input_card = ModernCard(main_container)
    input_card.pack(fill='x', pady=(0, 30))
    
    input_content = tk.Frame(input_card, bg=COLORS['bg_card'])
    input_content.pack(fill='x', padx=30, pady=30)
    
    # Input label
    tk.Label(
        input_content,
        text="Enter Test Case",
        font=('Inter', 16, 'normal'),
        bg=COLORS['bg_card'],
        fg=COLORS['text_primary']
    ).pack(anchor='w', pady=(0, 10))
    
    # Format hint with better styling
    format_frame = tk.Frame(input_content, bg=COLORS['accent_soft'])
    format_frame.pack(fill='x', pady=(0, 15))
    
    tk.Label(
        format_frame,
        text="Format: TCID, [number_of_steps], [1-step_description; 2-step_description; ...]",
        font=('Inter', 11),
        bg=COLORS['accent_soft'],
        fg=COLORS['text_secondary']
    ).pack(padx=15, pady=8)
    
    # Modern text input
    input_text = ModernEntry(input_content, height=6)
    input_text.pack(fill='x', pady=(0, 20))
    
    # Predict button
    predict_btn = ModernButton(
        input_content,
        text="Analyze Test Case",
        command=on_predict,
        bg_color=COLORS['accent_light'],
        hover_color=COLORS['accent_soft'],
        font=('Inter', 12, 'normal'),
        padding=(30, 15)
    )
    predict_btn.pack()

    # Result/error message area
    result_frame = tk.Frame(main_container, bg=COLORS['bg_primary'])
    result_frame.pack(fill='x')
    
    result_label = tk.Label(
        result_frame,
        text="",
        bg=COLORS['bg_primary'],
        justify='left',
        font=('Inter', 11)
    )
    result_label.pack(pady=10)



#------------------------------------------------------------------------------------------------
def show_order_page():
    global order_input_text, order_result_label  

    for widget in root.winfo_children():
        widget.destroy()
    root.configure(bg=COLORS['bg_primary'])

    # Main container
    main_container = tk.Frame(root, bg=COLORS['bg_primary'])
    main_container.pack(fill='both', expand=True, padx=40, pady=30)

    # Header frame with return button and title
    header_frame = tk.Frame(main_container, bg=COLORS['bg_primary'])
    header_frame.pack(fill='x', pady=(0, 30))

    # Return button on the left
    return_btn = ModernButton(
        header_frame,
        text="← Return",
        command=show_main_screen,
        bg_color=COLORS['order_page_hover'],
        hover_color=COLORS['order_page_btns'],
        font=('Inter', 10),
        padding=(16, 8)
    )
    return_btn.pack(side='left')

    # Centered title
    title_label = tk.Label(
        header_frame,
        text="Order Test Cases",
        font=('Inter', 24, 'normal'),
        bg=COLORS['bg_primary'],
        fg=COLORS['text_primary']
    )
    title_label.pack(expand=True)

    # Input section in a modern card
    input_card = ModernCard(main_container)
    input_card.pack(fill='x', pady=(0, 30))
    
    input_content = tk.Frame(input_card, bg=COLORS['bg_card'])
    input_content.pack(fill='x', padx=30, pady=30)
    
    # Input label
    tk.Label(
        input_content,
        text="Enter Multiple Test Cases",
        font=('Inter', 16, 'normal'),
        bg=COLORS['bg_card'],
        fg=COLORS['text_primary']
    ).pack(anchor='w', pady=(0, 10))
    
    # Format hint with better styling
    format_frame = tk.Frame(input_content, bg=COLORS['order_page_hover'])
    format_frame.pack(fill='x', pady=(0, 15))
    
    tk.Label(
        format_frame,
        text="Format for each line: TCID, [number_of_steps], [1-step_description; 2-step_description; ...]",
        font=('Inter', 11),
        bg=COLORS['order_page_hover'],
        fg=COLORS['text_secondary']
    ).pack(padx=15, pady=8)
    
    #Modern text input
    order_input_text = ModernEntry(input_content, height=6)
    order_input_text.pack(fill='x', pady=(0, 20))


    # Order button
    order_btn = ModernButton(
        input_content,
        text="Order Now",
        command=on_order_now,
        bg_color=COLORS['order_page_btns'],
        hover_color=COLORS['order_page_hover'],
        font=('Inter', 12, 'normal'),
        padding=(30, 15)
    )
    order_btn.pack()

    # Result/error message area (create once here)
    result_frame = tk.Frame(main_container, bg=COLORS['bg_primary'])
    result_frame.pack(fill='x')

    order_result_label = tk.Label(
        result_frame,
        text="",
        bg=COLORS['bg_primary'],
        justify='left',
        font=('Inter', 11),
        fg=COLORS['text_secondary'],
        wraplength = 500
    )
    order_result_label.pack(pady=10)



def on_order_now():
    global order_input_text, order_result_label
    global current_sort_mode, current_results

    input_raw = order_input_text.get("1.0", tk.END).strip()
    lines = [line.strip() for line in input_raw.split("\n") if line.strip()]
    
    results = []
    

    for line in lines:
        if validate_detailed_test_case(line):
            parts = line.strip().split(",", 2)
            if len(parts) != 3:
                continue  

            tcid = parts[0].strip()
            steps = [s.strip() for s in parts[2].split(";") if s.strip()]

            pred = parse_and_predict(line)
            if pred:
                pred["tcid"] = tcid
                pred["steps"] = steps
                pred["raw"] = line
                results.append(pred)
        else:
            order_result_label.config(text=f"Invalid format: {line}", fg="#ef4444")
            return

    

    if len(results) < 2:
        order_result_label.config(text="Please enter at least 2 valid test cases.", fg="#ef4444")
        return

    # Clear previous messages
    order_result_label.config(text="", fg=COLORS['text_secondary'])


    current_results = results


    # Display results below input box
    display_ordered_results(results)


def display_ordered_results(results):
    for widget in root.winfo_children():
        widget.destroy()

        global current_sort_mode

    # Compute priority scores if not already present
    for test in results:
        if 'priority_score' not in test:
            test['priority_score'] = (1 - test['predicted_passrate']) / test['predicted_duration']

    # Sort based on current sort mode
    if current_sort_mode == 0:
        results.sort(key=lambda x: x['priority_score'], reverse=True)
    elif current_sort_mode == 1:
        results.sort(key=lambda x: x['predicted_duration'])
    elif current_sort_mode == 2:
        results.sort(key=lambda x: x['predicted_passrate'])
    elif current_sort_mode == 3:
        results.sort(key=lambda x: x['tcid'])


    root.configure(bg=COLORS['bg_primary'])
    # Main container
    main_container = tk.Frame(root, bg=COLORS['bg_primary'])
    main_container.pack(fill='both', expand=True, padx=40, pady=30)
    # Header
    header_frame = tk.Frame(main_container, bg=COLORS['bg_primary'])
    header_frame.pack(fill='x', pady=(0, 30))
    return_btn = ModernButton(
        header_frame,
        text="← Return",
        command=show_order_page,
        bg_color=COLORS['order_page_hover'],
        hover_color=COLORS['order_page_btns'],
        font=('Inter', 10),
        padding=(16, 8)
    )
    return_btn.pack(side='left')
    title_label = tk.Label(
        header_frame,
        text="Ordered Test Cases",
        font=('Inter', 24, 'normal'),
        bg=COLORS['bg_primary'],
        fg=COLORS['text_primary']
    )
    title_label.pack(expand=True)
    

    # --- SORT BY DROPDOWN -----------------------------------------------------------------
    sort_container = tk.Frame(main_container, bg=COLORS['bg_primary'])
    sort_container.pack(fill='x', pady=(0, 10), padx=(0, 10))

    tk.Label(sort_container, text="", bg=COLORS['bg_primary']).pack(side='left', expand=True)

    # "Sort by" label with modern styling
    sort_label = tk.Label(
        sort_container,
        text="Sort by:",
        font=('Inter', 10),
        bg=COLORS['bg_primary'],
        fg=COLORS['text_secondary']
    )
    sort_label.pack(side='left', padx=(0, 8))

    # Create a frame to hold the dropdown with modern styling
    dropdown_frame = tk.Frame(sort_container, bg=COLORS['bg_card'], relief='flat', bd=0)
    dropdown_frame.pack(side='left', padx=1, pady=1)

    # Add subtle border
    border_frame = tk.Frame(dropdown_frame, bg=COLORS['border'], height=1)
    border_frame.pack(fill='x', side='bottom')

    # Configure modern combobox style
    style = ttk.Style()
    style.theme_use('clam')
    style.configure(
        "Modern.TCombobox",
        fieldbackground=COLORS['bg_card'],
        background=COLORS['bg_card'],
        foreground=COLORS['text_primary'],
        borderwidth=0,
        relief='flat',
        arrowcolor=COLORS['text_secondary'],
        font=('Inter', 10)
    )

    # Configure dropdown listbox styling
    style.configure(
        "Modern.TCombobox.Listbox",
        background=COLORS['bg_card'],
        foreground=COLORS['text_primary'],
        selectbackground=COLORS['accent_soft'],
        selectforeground=COLORS['text_primary'],
        borderwidth=1,
        relief='solid'
    )

    # Map hover and focus states
    style.map(
        "Modern.TCombobox",
        fieldbackground=[('focus', COLORS['accent_soft']), ('hover', COLORS['bg_secondary'])],
        background=[('focus', COLORS['accent_soft']), ('hover', COLORS['bg_secondary'])],
        bordercolor=[('focus', COLORS['accent_light'])]
    )

    # Dropdown menu with modern styling
    sort_labels = ["Priority", "Duration", "Pass Rate", "TCID"]
    sort_var = tk.StringVar(value=sort_labels[current_sort_mode])

    sort_dropdown = ttk.Combobox(
        dropdown_frame,
        textvariable=sort_var,
        values=["Priority", "Duration", "Pass Rate", "TCID"],
        state="readonly",
        width=12,
        style="Modern.TCombobox"
    )
    sort_dropdown.pack(padx=8, pady=6)

    
    # === CALLBACK FOR SORT CHANGE ===
    def on_sort_changed(event):
        global current_sort_mode

        selection = sort_var.get()
        if selection == "Priority":
            current_sort_mode = 0
        elif selection == "Duration":
            current_sort_mode = 1
        elif selection == "Pass Rate":
            current_sort_mode = 2
        elif selection == "TCID":
            current_sort_mode = 3
        else:
            current_sort_mode = 0  # fallback

        # Re-run ordering logic
        display_ordered_results(current_results)

    # Bind combobox selection change to callback
    sort_dropdown.bind("<<ComboboxSelected>>", on_sort_changed)



    # Add focus events for enhanced interaction
    def on_dropdown_focus_in(event):
        dropdown_frame.config(bg=COLORS['accent_soft'])

    def on_dropdown_focus_out(event):
        dropdown_frame.config(bg=COLORS['bg_card'])

    sort_dropdown.bind('<FocusIn>', on_dropdown_focus_in)
    sort_dropdown.bind('<FocusOut>', on_dropdown_focus_out)
# --- SORT BY DROPDOWN -----------------------------------------------------------------


    # Scrollable card - center it
    results_card = ModernCard(main_container)
    results_card.pack(fill='both', expand=True)
    
    # Create outer frame for centering
    outer_frame = tk.Frame(results_card, bg=COLORS['bg_card'])
    outer_frame.pack(fill='both', expand=True)
    
    canvas = tk.Canvas(outer_frame, bg=COLORS['bg_card'], highlightthickness=0)
    scrollbar = tk.Scrollbar(outer_frame, orient='vertical', command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg=COLORS['bg_card'])
    scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scrollable_frame, anchor='n')
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side='left', fill='both', expand=True)
    scrollbar.pack(side='right', fill='y')

    # Center the table horizontally
    center_container = tk.Frame(scrollable_frame, bg=COLORS['bg_card'])
    center_container.pack(expand=True)
    
    # Main table container
    table_container = tk.Frame(center_container, bg=COLORS['bg_card'])
    table_container.pack(pady=30)
    
    # Configure main table grid weights for responsive behavior
    headers = ["Order", "TCID", "Steps", "Duration", "Pass Rate"]
    column_weights = [0, 0, 1, 0, 0]  # Steps column gets all extra space
    
    for i, weight in enumerate(column_weights):
        table_container.columnconfigure(i, weight=weight, minsize=60)
    
    # Row colors for striping
    row_colors = [COLORS['bg_card'], COLORS.get('bg_secondary', '#f8f9fa')]
    
    # Store row frames for consistent height
    row_frames = []
    
    # === HEADERS ===
    header_frame = tk.Frame(table_container, bg=COLORS['bg_card'])
    header_frame.grid(row=0, column=0, columnspan=5, sticky='ew', padx=2, pady=1)
    
    for i, text in enumerate(headers):
        if i == 0:
            table_container.columnconfigure(i, minsize=80)
        elif i == 1:
            table_container.columnconfigure(i, minsize=120)
        elif i == 2:
            table_container.columnconfigure(i, minsize=400, weight=1)
        elif i == 3:
            table_container.columnconfigure(i, minsize=100)
        elif i == 4:
            table_container.columnconfigure(i, minsize=100)
            
        for i, text in enumerate(headers):
            if text == "Steps": # padding tuning just for title 'steps'
                custom_padx = 85
            elif text == "TCID": # padding tuning just for title 'TCID'
                custom_padx = 20
            else:
                custom_padx = 1

            header_label = tk.Label(
                header_frame,
                text=text,
                font=('Inter', 12, 'bold'),
                bg=COLORS['bg_card'],
                fg=COLORS['text_secondary'],
                anchor='w',
                padx=10,
                pady=8
            )
            header_label.grid(row=0, column=i, sticky='ew', padx=custom_padx)

    
    # Configure header frame columns
    for i in range(5):
        header_frame.columnconfigure(i, weight=column_weights[i])
    
    # === DATA ROWS ===
    for idx, item in enumerate(results, start=1):
        row_color = row_colors[idx % 2]
        
        # Create a frame for the entire row
        row_frame = tk.Frame(table_container, bg=row_color)
        row_frame.grid(row=idx, column=0, columnspan=5, sticky='ew', padx=2, pady=1)
        row_frames.append(row_frame)
        
        # Configure row frame columns to match table
        for i in range(5):
            row_frame.columnconfigure(i, weight=column_weights[i])
            if i == 0:
                row_frame.columnconfigure(i, minsize=80)
            elif i == 1:
                row_frame.columnconfigure(i, minsize=120)
            elif i == 2:
                row_frame.columnconfigure(i, minsize=400, weight=1)
            elif i == 3:
                row_frame.columnconfigure(i, minsize=100)
            elif i == 4:
                row_frame.columnconfigure(i, minsize=100)
        
        # === ORDER ===
        tk.Label(
            row_frame,
            text=str(idx),
            font=('Inter', 14, 'bold'),
            bg=row_color,
            fg=COLORS['order_page_btns'],
            anchor='center',
            padx=10,
            pady=8
        ).grid(row=0, column=0, sticky='ew', padx=1)
        
        # === TCID ===
        make_tcid_clickable(
            row_frame,
            item.get('tcid', '—'),
            item.get('raw', ''),
            on_return=lambda: display_ordered_results(current_results)
        )


        
        # === STEPS ===
        steps_text = ""
        for i, step in enumerate(item.get('steps', []), 1):
            clean_step = re.sub(r'^\d+-', '', step).strip()
            steps_text += f"{i}. {clean_step}\n"
        steps_text = steps_text.rstrip()
        
        steps_label = tk.Label(
            row_frame,
            text=steps_text,
            font=('Inter', 11),
            bg=row_color,
            fg=COLORS['text_primary'],
            anchor='nw',
            justify='left',
            wraplength=300,  # Will adjust based on available space
            padx=10,
            pady=8
        )
        steps_label.grid(row=0, column=2, sticky='ew', padx=1)
        
        # === DURATION ===
        tk.Label(
            row_frame,
            text=f"{item['predicted_duration']:.1f}" if 'predicted_duration' in item else "?",
            font=('Inter', 11),
            bg=row_color,
            fg=COLORS['text_primary'],
            anchor='center',
            padx=10,
            pady=8
        ).grid(row=0, column=3, sticky='ew', padx=1)
        
        # === PASS RATE ===
        tk.Label(
            row_frame,
            text=f"{item.get('predicted_passrate', '?')}%",
            font=('Inter', 11),
            bg=row_color,
            fg=COLORS['text_primary'],
            anchor='center',
            padx=10,
            pady=8
        ).grid(row=0, column=4, sticky='ew', padx=1)
    
    # Function to update wraplength and ensure consistent row heights
    def update_layout(event=None):
        # Get the current width of the table container
        table_container.update_idletasks()
        try:
            table_width = table_container.winfo_width()
            if table_width > 100:  # Ensure we have a reasonable width
                # Calculate available width for steps column
                available_width = max(250, int(table_width * 0.4))
                
                # Update wraplength for all steps labels and sync row heights
                for idx, row_frame in enumerate(row_frames):
                    steps_widget = None
                    for widget in row_frame.winfo_children():
                        if isinstance(widget, tk.Label) and widget.grid_info()['column'] == 2:
                            widget.configure(wraplength=available_width)
                            steps_widget = widget
                            break
                    
                    # Update row frame after wraplength change
                    if steps_widget:
                        row_frame.update_idletasks()
                        
        except:
            pass  # In case of any errors, just skip the update
    
    # Function to center the canvas content
    def center_canvas(event=None):
        canvas.update_idletasks()
        canvas_width = canvas.winfo_width()
        content_width = scrollable_frame.winfo_reqwidth()
        
        if content_width < canvas_width:
            # Center the content
            x_offset = (canvas_width - content_width) // 2
            canvas.create_window((x_offset, 0), window=scrollable_frame, anchor='n')
        else:
            # Content is wider than canvas, align to left
            canvas.create_window((0, 0), window=scrollable_frame, anchor='n')
    
    # Bind the update functions
    canvas.bind('<Configure>', center_canvas)
    root.after(100, update_layout)

    
    # Initial updates
    root.after(100, update_layout)
    root.after(150, center_canvas)
    

    tk.Frame(scrollable_frame, height=20, bg=COLORS['bg_card']).pack(fill='x')

    def on_mousewheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    canvas.bind_all("<MouseWheel>", on_mousewheel)
    canvas.focus_set()


    # === EXPORT SECTION ===
    export_container = ModernCard(scrollable_frame)
    export_container.pack(pady=30)

    export_inner = tk.Frame(export_container, bg=COLORS['bg_card'])
    export_inner.pack(padx=30, pady=20)

    # Export format label
    tk.Label(
        export_inner,
        text="Export as..",
        font=('Inter', 11),
        bg=COLORS['bg_card'],
        fg=COLORS['text_primary']
    ).grid(row=0, column=0, padx=(0, 10), pady=5)

    export_formats = ["TXT", "CSV", "PDF"]
    export_var = tk.StringVar(value=export_formats[0])

    # Export dropdown
    export_dropdown = ttk.Combobox(
        export_inner,
        textvariable=export_var,
        values=export_formats,
        state="readonly",
        width=10,
        style="Modern.TCombobox"
    )
    export_dropdown.grid(row=0, column=1, padx=(0, 20), pady=5)

    # Export button
    export_btn = ModernButton(
        export_inner,
        text="Export",
        command=lambda: export_results(results, export_var.get().upper()),
        bg_color=COLORS['accent_light'],
        hover_color=COLORS['accent_soft'],
        font=('Inter', 10),
        padding=(16, 8)
    )
    export_btn.grid(row=0, column=2, pady=5)

    # Center all elements in the row
    export_inner.columnconfigure(0, weight=0)
    export_inner.columnconfigure(1, weight=0)
    export_inner.columnconfigure(2, weight=0)




def show_main_screen():
    for widget in root.winfo_children():
        widget.destroy()
    run_main_screen()


def open_result_popup(test_input):
    popup = tk.Toplevel(root)
    popup.title("Prediction Result")
    popup.geometry("960x640")
    popup.configure(bg=COLORS['bg_primary'])

    show_result_page(test_input, parent=popup)


def make_tcid_clickable(parent, tcid, raw_data, on_return):
    def open_result():
        show_result_page(raw_data, on_return=on_return)

    label = tk.Label(
        parent,
        text=f"{tcid} ↗",
        font=('Inter', 12, 'underline'),
        bg=parent['bg'],
        fg='#0000B3',
        cursor='hand2',
        padx=10,
        pady=8,
        anchor='w'
    )
    label.grid(row=0, column=1, sticky='ew', padx=1)
    label.bind("<Button-1>", lambda e: open_result())


def export_results(results, format):

    # --- Determine sort label ---
    sort_labels = {
        0: "Priority",
        1: "Duration",
        2: "Pass Rate",
        3: "TCID"
    }
    sorted_by = sort_labels.get(current_sort_mode, "Unknown")

    # --- Create default filename ---
    today = datetime.now().strftime('%Y-%m-%d_%H-%M')
    default_filename = f"TC_Comparisons_{today}.{format.lower()}"

    # --- Ask for save location ---
    filetypes = {
        "TXT": [("Text files", "*.txt")],
        "CSV": [("CSV files", "*.csv")],
        "PDF": [("PDF files", "*.pdf")]
    }
    file_path = filedialog.asksaveasfilename(
        defaultextension=f".{format.lower()}",
        filetypes=filetypes.get(format.upper(), [("All files", "*.*")]),
        initialfile=default_filename
    )
    if not file_path:
        return

    # --- TXT Export ---
    if format == "TXT":
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"SORTED BY: {sorted_by.upper()}\n\n")
            for r in results:
                f.write(f"TCID: {r['tcid']}\n")
                f.write(f"Duration: {r['predicted_duration']}s\n")
                f.write(f"Pass Rate: {r['predicted_passrate']}%\n")
                f.write("Steps:\n")
                for step in r.get('steps', []):
                    f.write(f"  - {step}\n")
                f.write("\n")

    # --- CSV Export ---
    elif format == "CSV":
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([f"SORTED BY: {sorted_by.upper()}"])
            writer.writerow(["TCID", "Duration", "Pass Rate", "Steps"])
            for r in results:
                steps_joined = " | ".join(r.get('steps', []))
                writer.writerow([r['tcid'], r['predicted_duration'], r['predicted_passrate'], steps_joined])

    # --- PDF Export ---
    elif format == "PDF":
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, txt=f"SORTED BY: {sorted_by.upper()}", ln=True)
        pdf.ln(4)

        pdf.set_font("Arial", "", 10)
        for r in results:
            pdf.cell(200, 10, txt=f"TCID: {r['tcid']}", ln=True)
            pdf.cell(200, 10, txt=f"Duration: {r['predicted_duration']}s", ln=True)
            pdf.cell(200, 10, txt=f"Pass Rate: {r['predicted_passrate']}%", ln=True)
            pdf.multi_cell(0, 10, txt="Steps:\n" + "\n".join(r.get('steps', [])))
            pdf.ln()

        pdf.output(file_path)

    else:
        print("Unsupported format")



# Initialize the application
run_main_screen()
root.mainloop()

