import tkinter as tk
from tkinter import ttk
import re

# Create main window
root = tk.Tk()
root.title("AI Test Case Predictor")
root.geometry("700x500")
root.configure(bg="white")

def show_result_page(test_input):
    # === Font Definitions ===
    main_font = ("Arial", 15)
    text_font = ("Arial", 13)

    # Clear existing content
    for widget in root.winfo_children():
        widget.destroy()

    root.configure(bg="white")

    # === Title ===
    title_label = tk.Label(root, text="Prediction Results\n", font=("Arial", 20, "bold"), bg="white")
    title_label.grid(row=0, column=1, pady=(30, 20), sticky="n")

    # === Main content frame (contains left and right columns) ===
    content_frame = tk.Frame(root, bg="white")
    content_frame.grid(row=1, column=1, padx=20, sticky="n")
    


    # Make columns expandable
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=0)
    root.grid_columnconfigure(2, weight=1)

    # === Left Column ===
    left_column = tk.Frame(content_frame, bg="white")
    left_column.grid(row=0, column=0, sticky="nw")

    # User-entered test case
    test_case_label = tk.Label(
        left_column,
        text=f"Test Case: {test_input}",
        font=main_font,
        bg="white",
        anchor="w",
        justify="left",
        wraplength=600
    )
    test_case_label.grid(row=0, column=0, sticky="w", pady=(15, 35))

    # Estimated duration
    duration_label = tk.Label(
        left_column,
        text="Estimated Duration: ...",
        font=main_font,
        bg="white",
        anchor="w",
        justify="left"
    )
    duration_label.grid(row=1, column=0, sticky="w", pady=(0, 12))

    # Estimated pass rate
    passrate_label = tk.Label(
        left_column,
        text="Estimated Pass Rate: ...",
        font=main_font,
        bg="white",
        anchor="w",
        justify="left"
    )
    passrate_label.grid(row=2, column=0, sticky="w")


    # === Right Column (Similar Test Cases) ===
    right_column = tk.Frame(content_frame, bg="#e6f2ff", bd=1, relief="solid", width=250, height=640)
    right_column.grid(row=0, column=1, padx=(40, 20), sticky="ne")
    right_column.grid_propagate(False)

    sim_title = tk.Label(
        right_column,
        text="Similar Test Cases",
        font=main_font,
        bg="#e6f2ff"
    )
    sim_title.pack(anchor="w", pady=(10, 10), padx=10)

    for i in range(3):
        sim = tk.Label(
            right_column,
            text=f"• Similar TestCase #{i+1}",
            font=text_font,
            bg="#e6f2ff",
            anchor="w",
            justify="left",
            wraplength=240
        )
        sim.pack(anchor="w", pady=2, padx=10)
    tk.Label(right_column, text="", bg="#e6f2ff").pack(expand=True, fill="both")

    # === Placeholder for ML model results ===
    predicted_duration = "18.7 seconds"
    predicted_passrate = "64%"

    duration_label.config(text=f"Estimated Duration: {predicted_duration}")
    passrate_label.config(text=f"Estimated Pass Rate: {predicted_passrate}")



#validating entered test case format
def validate_detailed_test_case(line: str) -> bool:
    try:
        # Strip and split into 5 components
        parts = [p.strip() for p in line.strip().rsplit(",", 2)]
        if len(parts) != 3:
            return False

        # Extract: [step_part before last 2 commas], duration, result
        pre, duration_str, result = parts
        pre_parts = pre.split(",", 2)
        if len(pre_parts) != 3:
            return False

        tcid, num_steps_str, step_desc = pre_parts
        if not tcid or not num_steps_str.isdigit():
            return False

        num_steps = int(num_steps_str)

        # Count actual steps by detecting N- pattern (e.g., 1-, 2-, 3-)
        found_steps = re.findall(r'\b\d+-', step_desc)
        if len(found_steps) != num_steps:
            return False

        # Duration must be a valid float
        float(duration_str)  # will raise ValueError if invalid

        # Result must be PASS or FAIL
        if result not in {"PASS", "FAIL"}:
            return False

        return True
    except:
        return False


# === Function triggered on Predict button click ===
def on_predict():
    test_input = input_text.get("1.0", tk.END).strip()

    if not test_input:
        result_label.config(text="Please enter a test case first.")
        return

    else:
        if(True):  #debug: actual condition -> validate_detailed_test_case(test_input)
             # Format is valid → clear everything from screen
            for widget in root.winfo_children():
                widget.destroy()
            show_result_page(test_input)
            # TODO: Call your ML model here using test_input********************************************************
        
        else:
            result_label.config(text="Invalid format!")
            


# === Input frame ===
input_frame = tk.Frame(root, bg="white",  pady=10)
input_frame.pack(fill="x")

input_label = tk.Label(
    input_frame,
    text="Enter your test case in following format:",
    bg="white",
    anchor="w",
    justify="left",
    font=("Arial", 20, "bold")
)
input_label.pack( padx=20, pady=20)

format_hint = tk.Label(
    input_frame,
    text="TCID, [n_num_of_steps], [1-stepnumber]; [n-stepnumber] , duration (seconds), result (PASS/FAIL)",
    bg="white",
    anchor="w",
    justify="left",
    fg="gray",
    font=("Arial", 12, "italic")
)
format_hint.pack(padx=20, pady=30)

input_text = tk.Text(input_frame, height=5, width=80, borderwidth=1, relief="solid")
input_text.pack(pady=5)

predict_button = tk.Button(root, text="Predict", command=on_predict)
predict_button.pack(pady=10)

# === Output frame ===
output_frame = tk.Frame(root, bg="white", padx=10, pady=10)
output_frame.pack(fill="both", expand=True)

# Result label
result_label = tk.Label(output_frame, text="", bg="white", justify="left", fg="#C73333", font=("Arial", 11))
result_label.pack(padx = 10)



# Run the GUI loop
root.mainloop()





#kod deposu
"""
# === Canvas-based pie chart for pass rate ===
    pass_rate = 42  # örnek yüzde, daha sonra model çıktısına bağlanacak

    canvas = tk.Canvas(left_column, width=150, height=150, bg="white", highlightthickness=0)
    canvas.grid(row=3, column=0, pady=(20, 10), sticky="w")  # passrate_label'ın altına

    x0, y0, x1, y1 = 10, 10, 140, 140

    # PASS (green slice)
    canvas.create_arc(x0, y0, x1, y1, start=90, extent=-pass_rate * 3.6, fill="#CCFFCC")

    # FAIL (red slice)
    canvas.create_arc(x0, y0, x1, y1, start=90 - pass_rate * 3.6, extent=-(360 - pass_rate * 3.6), fill="#D04343")

    # Optional: add labels
    canvas.create_text(75, 75, text=f"{pass_rate}%", fill="white", font=("Arial", 12, "bold"))
"""