import tkinter as tk
from tkinter import ttk
import re
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained ML models and preprocessors
with open("clf.pkl", "rb") as f:
    clf = pickle.load(f)
with open("reg.pkl", "rb") as f:
    reg = pickle.load(f)
with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("X_features.pkl", "rb") as f:
    X_features = pickle.load(f)
with open("test_ids.pkl", "rb") as f:
    test_ids = pickle.load(f)
with open("similar_case_details.pkl", "rb") as f:
    similar_case_details = pickle.load(f)

# Modern Color Palette - Soft Pastels
COLORS = {
    'bg_primary': '#f8fafc',      # Very light blue-gray
    'bg_secondary': '#f1f5f9',    # Light blue-gray
    'bg_card': '#ffffff',         # Pure white
    'accent_soft': '#e0e7ff',     # Soft lavender
    'accent_light': '#c7d2fe',    # Light indigo
    'text_primary': '#1e293b',    # Dark slate
    'text_secondary': '#64748b',  # Medium slate
    'text_muted': '#94a3b8',      # Light slate
    'success': '#dcfce7',         # Soft green
    'warning': '#fef3c7',         # Soft yellow
    'error': '#fee2e2',           # Soft red
    'border': '#e2e8f0',          # Light border
    'shadow': '#f1f5f9'           # Subtle shadow
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

def show_result_page(test_input):
    root.configure(bg=COLORS['bg_primary'])
    
    # ML Prediction Results
    results = parse_and_predict(test_input)
    if not results:
        print("Prediction failed.")
        return
    
    # Main container with padding
    main_container = tk.Frame(root, bg=COLORS['bg_primary'])
    main_container.pack(fill='both', expand=True, padx=30, pady=20)
    
    # Header section
    header_frame = tk.Frame(main_container, bg=COLORS['bg_primary'])
    header_frame.pack(fill='x', pady=(0, 30))
    
    # Return button - modern style
    return_btn = ModernButton(
        header_frame, 
        text="← Return", 
        command=show_main_screen,
        bg_color=COLORS['bg_secondary'],
        hover_color=COLORS['accent_soft'],
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

def show_main_screen():
    for widget in root.winfo_children():
        widget.destroy()
    run_main_screen()

# Initialize the application
run_main_screen()
root.mainloop()