import tkinter as tk
from tkinter import ttk

from mrs import *

user_item_matrix, movie_db, ratings = load_data()
user_id, user_ratings, = None, None


def login(event=None):
    global user_id
    user_id = int(user_id_entry.get())

    if check_user_id(user_id):
        login_window.destroy()
        rec_system(user_id)
    else:
        error_label.config(text='Invalid User ID')


l_width, l_height = 480, 360
login_window = tk.Tk()
login_window.title('Movie Recommendation System')

l_screen_width = login_window.winfo_screenwidth()
l_screen_height = login_window.winfo_screenheight()

lx = (l_screen_width - l_width) // 2
ly = (l_screen_height - l_height) // 2

login_window.geometry(f"{l_width}x{l_height}+{lx}+{ly}")

logo_image = tk.PhotoImage(file='logo.png')
logo_image = logo_image.subsample(5, 5)
logo_label = ttk.Label(login_window, image=logo_image)
logo_label.pack(pady=10)

user_id_label = ttk.Label(login_window, text='User ID')
user_id_label.pack(pady=10)

user_id_entry = ttk.Entry(login_window)
user_id_entry.pack()

error_label = ttk.Label(login_window, text='', foreground='red')
error_label.pack()

login_window.bind('<Return>', login)
login_button = ttk.Button(login_window, text='Login', command=login)
login_button.pack(pady=10)


def rec_system(user_id):
    def print_table(recommendations, similar=False):
        if similar:
            rec_sim_lable.config(
                text=f"Similar Movies to '{movie_entry.get()}'")
        else:
            rec_sim_lable.config(text='Recommended Movies')

        movie_table.delete(*movie_table.get_children())
        for _, row in recommendations.iterrows():
            movie_table.insert('', tk.END, values=(
                row['title'], row['genres'], row['rating']))

    def show_recommended_movies():
        global user_ratings

        correlations = pearson_corr(user_id)
        recommendations, user_ratings = recommend_movies(correlations, 15)
        print_table(recommendations)

    def refresh_movies():
        print('Refreshing...')
        show_recommended_movies()
        print('Refreshed')

    def show_similar_movies(selected_movie):
        similarites = jaccard_sim(selected_movie)
        recommendations = recommend_similar_movies(user_id, similarites, 15)
        print_table(recommendations, similar=True)

    def search_results(event=None):
        search_result_label.config(text='You might be looking for')
        movie_titles = movie_db['title']
        search_query = movie_entry.get().lower()
        filtered_movies = [
            movie for movie in movie_titles if search_query in movie.lower()]

        movie_listbox.delete(0, tk.END)
        for movie in filtered_movies:
            movie_listbox.insert(tk.END, movie)

    def select_movie(event):
        movie_entry.delete(0, tk.END)
        selected_movie = movie_listbox.get(tk.ACTIVE)

        movie_entry.insert(tk.END, selected_movie)
        show_similar_movies(selected_movie)

    def rate_movie():
        def save_rating():
            global user_ratings
            try:
                rating = float(rating_entry.get())
                if rating < 1 or rating > 5:
                    raise ValueError()

                user_ratings = update_rating(
                    user_id, movie_title, rating, user_ratings)

                print(f'Rated \'{movie_title}\' {rating}')
                rating_window.destroy()
            except (IndexError, ValueError):
                label = ttk.Label(rating_window, text='', foreground='red')
                label.pack()
                label.config(text='Invalid rating!')

        selected_movie = movie_table.selection()
        if not selected_movie:
            return

        movie_info = movie_table.item(selected_movie)
        movie_title = movie_info['values'][0]

        rating_window = tk.Toplevel(window)
        rating_window.title("Rate Movie")

        r_screen_width = rating_window.winfo_screenwidth()
        r_screen_height = rating_window.winfo_screenheight()

        rating_label = ttk.Label(rating_window, text=f"'{movie_title}'")
        rating_label.pack(pady=10)

        rating_entry = ttk.Entry(rating_window, width=10)
        rating_entry.pack(pady=10)

        submit_button = ttk.Button(
            rating_window, text="Submit", command=lambda: save_rating())
        submit_button.pack(pady=10)

        r_width, r_height = 400, 180
        rx = (r_screen_width - r_width) // 2
        ry = (r_screen_height - r_height) // 2
        rating_window.geometry(f"{r_width}x{r_height}+{rx}+{ry}")

        rating_window.mainloop()

    def stats():
        global user_ratings

        if not isinstance(user_ratings, pd.DataFrame):
            return

        rated_movs = user_ratings.dropna(subset=['actual_rating'], axis=0)
        if rated_movs.empty:
            return

        # print(rated_movs)
        eval_performance(rated_movs)
        user_ratings = None

    window = tk.Tk()
    s_width, s_height = 720, 500
    window.title(f'Movie Recommendation System (User {user_id})')

    search_label = ttk.Label(window, text='Search Movies')
    search_label.pack(pady=10)

    movie_entry = ttk.Entry(window, width=50)
    movie_entry.pack()
    movie_entry.bind("<KeyRelease>", search_results)
    # movie_entry.bind('<Return>', show_similar_movies)

    search_result_label = ttk.Label(window, text='')
    search_result_label.pack(pady=10)

    movie_listbox = tk.Listbox(window, width=50, height=5)
    movie_listbox.pack()
    movie_listbox.bind("<<ListboxSelect>>", select_movie)

    button_frame = tk.Frame(window)
    button_frame.pack(pady=6)

    statistics_button = ttk.Button(
        button_frame, text="Statistics", command=stats)
    statistics_button.pack(side=tk.LEFT, padx=5)

    refresh_button = ttk.Button(
        button_frame, text="Refresh", command=refresh_movies)
    refresh_button.pack(side=tk.LEFT, padx=5)

    rec_sim_lable = ttk.Label(window, text='Recommended Movies')
    rec_sim_lable.pack(pady=10)
    movie_table = ttk.Treeview(window, columns=(
        'Movie', 'Genres', 'Rating'), show='headings', height=20)

    movie_table.bind("<Double-1>", lambda e: rate_movie())

    movie_table.heading('Movie', text='Movie')
    movie_table.heading('Rating', text='Rating')
    movie_table.heading('Genres', text='Genres')
    movie_table.pack(pady=10)

    movie_table.column('Movie', width=300, anchor='center')
    movie_table.column('Genres', width=300, anchor='center')
    movie_table.column('Rating', width=100, anchor='center')

    m_screen_width = window.winfo_screenwidth()
    m_screen_height = window.winfo_screenheight()

    mx = (m_screen_width - s_width) // 2
    my = (m_screen_height - s_height) // 2

    window.geometry(f"{s_width}x{s_height}+{mx}+{my}")

    show_recommended_movies()
    window.mainloop()


login_window.mainloop()
