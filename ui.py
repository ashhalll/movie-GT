import customtkinter as ctk
from predictor import recommend_movies

# Function to start the UI and get recommendations
def start_ui(prediction_matrix, movies_df):
    # Initialize the app window
    app = ctk.CTk()
    app.geometry('800x600')
    app.title('Movie Recommendation System')

    def get_recommendations():
        user_id = int(entry_user_id.get())
        top_n = int(entry_top_n.get())
        recommendations = recommend_movies(user_id, top_n)
        result_label.configure(text=str(recommendations))

    # Create frame for widgets
    frame = ctk.CTkFrame(master=app)
    frame.pack(pady=20, padx=20, fill='both', expand=True)

    # Add widgets (labels, entry fields, button, result display)
    label = ctk.CTkLabel(frame, text="Enter User ID and Number of Recommendations:")
    label.pack(pady=10)

    entry_user_id = ctk.CTkEntry(frame, placeholder_text="User ID")
    entry_user_id.pack(pady=5)

    entry_top_n = ctk.CTkEntry(frame, placeholder_text="Top N Recommendations")
    entry_top_n.pack(pady=5)

    button = ctk.CTkButton(frame, text="Get Recommendations", command=get_recommendations)
    button.pack(pady=10)

    result_label = ctk.CTkLabel(frame, text="", justify="left", anchor="w", wraplength=700)
    result_label.pack(pady=40)

    # Start the main loop to display the GUI
    app.mainloop()

# If this file is run directly, start the UI
if __name__ == "__main__":
    # Placeholder for prediction matrix and movies dataframe
    # Replace with actual loaded data when running
    start_ui(None, None)