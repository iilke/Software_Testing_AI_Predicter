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
    similar_case_details = pickle.load(f)  # { test_id: {steps, duration, result} }



# Create main window
root = tk.Tk()
root.title("AI Test Case Predictor")
root.geometry("700x500")
root.configure(bg="white")


def show_result_page(test_input):
    root.configure(bg="white")

    # === ML Prediction Results ===
    results = parse_and_predict(test_input)
    if not results:
        print("Prediction failed.")
        return

    # === Top Frame for Return Button ===
    top_frame = tk.Frame(root, bg="white")
    top_frame.grid(row=0, column=0, columnspan=3, sticky="ew")

    return_button = tk.Button(top_frame, text="⟵ Return", font=("Arial", 10), command=show_main_screen)
    return_button.pack(side="left", padx=20, pady=20)

    # === Font Definitions ===
    main_font = ("Arial", 15)
    text_font = ("Arial", 13)

    # === Title ===
    title_label = tk.Label(root, text="Prediction Results\n", font=("Arial", 20, "bold"), bg="white")
    title_label.grid(row=0, column=1, pady=(30, 20), sticky="n")

    # === Main content frame ===
    content_frame = tk.Frame(root, bg="white")
    content_frame.grid(row=1, column=1, padx=20, sticky="n")

    # Make columns expandable
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=0)
    root.grid_columnconfigure(2, weight=1)

    # === Left Column ===
    left_column = tk.Frame(content_frame, bg="white")
    left_column.grid(row=0, column=0, sticky="nw")

    # === Parse test input ===
    try:
        parts = test_input.strip().split(",", 2)
        tcid = parts[0].strip()
        steps_raw = parts[2].strip()
        step_lines = [s.strip() for s in steps_raw.split(";") if s.strip()]
    except:
        tcid = "Unknown"
        step_lines = ["Parsing error"]

    # === Test Case Breakdown ===
    tk.Label(left_column, text="Test Case:", font=main_font, bg="white").grid(row=0, column=0, sticky="w", pady=(15, 5))
    tk.Label(left_column, text=f"TCID: {tcid}", font=text_font, bg="white").grid(row=1, column=0, sticky="w", pady=(0, 10))
    tk.Label(left_column, text="Step Descriptions:", font=text_font, bg="white").grid(row=2, column=0, sticky="w")

    for idx, step in enumerate(step_lines):
        tk.Label(
            left_column,
            text=f"    {step}",
            font=("Arial", 12),
            bg="white",
            anchor="w",
            justify="left",
            wraplength=580
        ).grid(row=3 + idx, column=0, sticky="w", pady=1)

    # === Estimated Duration and Pass Rate ===
    duration_label = tk.Label(
        left_column,
        text=f"Estimated Duration: {results['predicted_duration']} seconds",
        font=main_font,
        bg="white",
        anchor="w",
        justify="left"
    )
    duration_label.grid(row=3 + len(step_lines), column=0, sticky="w", pady=(20, 12))

    passrate_label = tk.Label(
        left_column,
        text=f"Estimated Pass Rate: {results['predicted_passrate']}%",
        font=main_font,
        bg="white",
        anchor="w",
        justify="left"
    )
    passrate_label.grid(row=4 + len(step_lines), column=0, sticky="w")

    # === Right Column: Similar Test Cases ===
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

    for case_id in results["similar_cases"]:
        detail = similar_case_details.get(case_id, {})
        steps = detail.get("steps", "N/A")
        duration = detail.get("duration", "N/A")
        result = detail.get("result", "N/A")

        info = f"• {case_id}\nDuration: {duration}s\nResult: {result}\nSteps: {steps}"
        
        tk.Label(
            right_column,
            text=info,
            font=("Arial", 11),
            bg="#e6f2ff",
            anchor="w",
            justify="left",
        wraplength=230
    ).pack(anchor="w", pady=(5, 10), padx=10)


    # Spacer to fill remaining space
    tk.Label(right_column, text="", bg="#e6f2ff").pack(expand=True, fill="both")






#validating entered test case format
def validate_detailed_test_case(line: str) -> bool:
    try:
        # Strip and split into 3 parts: TCID, num_steps, steps
        parts = [p.strip() for p in line.strip().split(",", 2)]
        if len(parts) != 3:
            return False

        tcid, num_steps_str, step_desc = parts

        if not tcid or not num_steps_str.isdigit():
            return False

        num_steps = int(num_steps_str)

        # Count actual steps using pattern like '1-', '2-', etc.
        found_steps = re.findall(r'\b\d+-', step_desc)
        if len(found_steps) != num_steps:
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

    if validate_detailed_test_case(test_input):
        for widget in root.winfo_children():
            widget.destroy()
        show_result_page(test_input)
    else:
        result_label.config(text="Invalid format!")



def parse_and_predict(test_input):
    """
    Parses user input in format:
    TCID, num_steps, 1-step; 2-step; ...
    Returns predicted duration, pass rate, and similar cases.
    """
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

        # Extract and clean step descriptions
        found_steps = re.findall(r'\b\d+-', steps_raw)
        if len(found_steps) != num_steps:
            return None

        # Remove step numbers like "1-", "2-", etc.
        step_keywords = re.sub(r'\b\d+-', '', steps_raw).strip()

        # Vectorize
        X_keywords = tfidf.transform([step_keywords]).toarray()
        X_steps = scaler.transform([[num_steps]])
        X_input = np.hstack([X_keywords, X_steps])

        # Predictions
        pred_duration = reg.predict(X_input)[0]
        pred_passrate = clf.predict_proba(X_input)[0][1] * 100

        #similar test cases    
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

    # === Input frame ===
    input_frame = tk.Frame(root, bg="white", pady=10)
    input_frame.pack(fill="x")

    input_label = tk.Label(
        input_frame,
        text="Enter your test case in following format:",
        bg="white",
        anchor="w",
        justify="left",
        font=("Arial", 20, "bold")
    )
    input_label.pack(padx=20, pady=20)

    format_hint = tk.Label(
        input_frame,
        text="TCID, [n_num_of_steps], [1-stepnumber]; [n-stepnumber]",
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
    result_label.pack(padx=10)


def show_main_screen():
    for widget in root.winfo_children():
        widget.destroy()
    run_main_screen()


run_main_screen()
# Run the GUI loop
root.mainloop()

